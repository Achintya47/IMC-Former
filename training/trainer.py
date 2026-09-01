"""
IMC-Former Trainer
==================
Full training loop with:
  - Curriculum learning (sampler rebuilt each epoch)
  - Per-batch and per-epoch logging (console + file)
  - Validation every epoch; test and generalization splits on final model
  - Best model selection on val FPR (primary) subject to val accuracy floor
  - TensorBoard logging for all metrics and losses
  - Checkpoint saving (best + periodic)
  - Gradient clipping
  - Differential learning rates (encoder vs heads)
  - BUG-FIXED (v2): NaN/Inf-safe training step. A non-finite loss or gradient
    no longer corrupts the model. The offending batch's Taskset_IDs are
    logged, the optimizer step is skipped, and training continues cleanly.
"""

import copy
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from configs.config import IMCFormerConfig
from data.dataset import build_dataloaders, update_train_sampler
from evaluation.metrics import (
    accumulate_predictions,
    compute_metrics,
    compute_metrics_by_n,
)
from models.imc_former import IMCFormer
from models.loss import IMCLoss


# ── Logging setup ──────────────────────────────────────────────────────────────

def setup_logging(log_dir: str, run_name: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("imc_former")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(log_dir, f"{run_name}.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ── NEW: NaN/Inf diagnosis helper ───────────────────────────────────────────

def _first_nonfinite_tensor_name(batch: Dict) -> Optional[str]:
    """Return the name of the first batch tensor containing a non-finite value."""
    for key in ("features", "mask", "context", "labels"):
        t = batch.get(key)
        if isinstance(t, torch.Tensor) and not torch.isfinite(t).all():
            return key
    return None


# ── Trainer ────────────────────────────────────────────────────────────────────

class IMCFormerTrainer:
    """
    End-to-end trainer for IMC-Former.

    Usage:
        trainer = IMCFormerTrainer(cfg)
        trainer.train()
        trainer.evaluate_generalization()
    """

    def __init__(self, cfg: IMCFormerConfig):
        self.cfg = cfg
        tcfg  = cfg.training
        mcfg  = cfg.model
        lcfg  = cfg.loss

        torch.manual_seed(tcfg.seed)
        if torch.cuda.is_available() and tcfg.device == "cuda":
            torch.cuda.manual_seed_all(tcfg.seed)

        self.device = torch.device(tcfg.device if torch.cuda.is_available() else "cpu")

        # ── Logging & TensorBoard ──────────────────────────────────────────────
        os.makedirs(tcfg.log_dir,        exist_ok=True)
        os.makedirs(tcfg.checkpoint_dir, exist_ok=True)
        os.makedirs(tcfg.figure_dir,     exist_ok=True)

        self.logger = setup_logging(tcfg.log_dir, tcfg.run_name)
        self.writer = SummaryWriter(
            log_dir=os.path.join(tcfg.log_dir, "tensorboard", tcfg.run_name)
        )

        # ── Data ────────────────────────────────────────────────────────────────
        # NEW: generalization splits (gen_small..gen_xxlarge) are NOT loaded
        # here. build_dataloaders returns LazyGenSplit handles by default
        # (eager_gen=False) -- each split's IMCDataset is only actually built
        # inside evaluate_generalization(), evaluated, and freed before the
        # next split is loaded. Previously all 5 splits were built eagerly
        # alongside train/val/test, paying their full memory cost for the
        # entire duration of training even though they're only used once, at
        # the very end. This was a major contributor to the OOM kill once the
        # in-distribution corpus also grew into the millions of rows.
        self.logger.info("Loading datasets (train/val/test only; "
                         "generalization splits load lazily at eval time) ...")
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.gen_items,
            self.norm_stats,
        ) = build_dataloaders(cfg, eager_gen=False)
        self.logger.info(
            f"  Train: {len(self.train_loader.dataset):,} | "
            f"Val: {len(self.val_loader.dataset):,} | "
            f"Test: {len(self.test_loader.dataset):,}"
        )

        # ── Model ────────────────────────────────────────────────────────────────
        self.model = IMCFormer(mcfg).to(self.device)
        self.logger.info(
            f"IMC-Former: {self.model.n_parameters:,} trainable parameters"
        )

        # ── Loss ─────────────────────────────────────────────────────────────────
        pos_weights = self.train_loader.dataset.get_scheduler_class_weights()
        self.logger.info(f"  BCE positive class weights: {pos_weights.tolist()}")

        self.criterion = IMCLoss(
            cfg=lcfg,
            scheduler_names=mcfg.scheduler_names,
            pos_weights=pos_weights,
        ).to(self.device)

        # ── Optimizer (differential LR) ──────────────────────────────────────────
        encoder_params = (
            list(self.model.task_encoder.parameters())
            + list(self.model.lo_transformer.parameters())
            + list(self.model.hi_transformer.parameters())
            + list(self.model.cross_hi_on_lo.parameters())
            + list(self.model.cross_lo_on_hi.parameters())
            + list(self.model.lo_pool.parameters())
            + list(self.model.hi_pool.parameters())
            + list(self.model.context_mlp.parameters())
        )
        head_params = list(self.model.heads.parameters())

        self.optimizer = AdamW(
            [
                {"params": encoder_params, "lr": tcfg.encoder_lr},
                {"params": head_params,    "lr": tcfg.head_lr},
            ],
            weight_decay=tcfg.weight_decay,
        )

        # ── LR Scheduler ─────────────────────────────────────────────────────────
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=tcfg.lr_factor,
            patience=tcfg.lr_patience,
            min_lr=tcfg.lr_min,
        )

        # ── State ────────────────────────────────────────────────────────────────
        self.best_val_fpr  = float("inf")
        self.best_val_acc  = 0.0
        self.best_epoch    = 0
        self.no_improve    = 0
        self.history: List[Dict] = []

        # ── NEW: NaN/Inf incident tracking ──────────────────────────────────────
        # A pre-existing snapshot of the last known-good model/optimizer state is
        # kept in memory so a corrupted step can be rolled back immediately,
        # instead of relying solely on skipping the optimizer step (which
        # protects future steps but does nothing for a step that already wrote
        # NaN into the weights via some path other than the ones we guard here).
        self.nan_incidents: List[Dict] = []
        self._last_good_state = None  # populated after the first clean batch

        # ── NEW: instability auto-recovery ──────────────────────────────────────
        # A single non-finite batch (guards 1-4 in _train_epoch) is normal and
        # is simply skipped -- almost always a one-off malformed data row. But
        # if several batches in a ROW are all non-finite regardless of which
        # Taskset_IDs they contain, that is a categorically different signal:
        # the model's own weights have drifted into a numerically unstable
        # region (e.g. one parameter's magnitude has grown large enough that
        # backward() overflows for essentially any input, even though forward
        # activations stay bounded thanks to LayerNorm). Skipping the
        # optimizer step protects against further corruption, but does NOT
        # undo the drift that got here -- left alone, every subsequent batch
        # for the rest of training hits the same wall and no learning happens
        # again. `consecutive_nonfinite` tracks the current streak; once it
        # crosses `nonfinite_recovery_patience`, `_recover_from_instability`
        # rolls the model AND optimizer back to the last clean snapshot and
        # halves the learning rate, then keeps training instead of stalling
        # for the rest of the run.
        self.consecutive_nonfinite = 0
        self.nonfinite_recovery_patience = getattr(
            tcfg, "nonfinite_recovery_patience", 5
        )
        self.lr_reduction_factor_on_instability = getattr(
            tcfg, "lr_reduction_factor_on_instability", 0.5
        )
        self.recovery_events: List[Dict] = []

        self.logger.info(
            f"Training on {self.device} | "
            f"Encoder LR={tcfg.encoder_lr} | Head LR={tcfg.head_lr} | "
            f"λ={lcfg.fp_penalty_lambda} | μ={lcfg.hi_fp_penalty_mu}"
        )

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self):
        tcfg = self.cfg.training
        self.logger.info(f"Starting training for {tcfg.num_epochs} epochs")

        for epoch in range(1, tcfg.num_epochs + 1):
            self.train_loader = update_train_sampler(self.train_loader, epoch, self.cfg)

            epoch_start = time.time()
            train_metrics = self._train_epoch(epoch)
            val_metrics   = self._evaluate(self.val_loader, split="val")
            epoch_time    = time.time() - epoch_start

            val_fpr = val_metrics["fpr/avg"]
            val_acc = val_metrics["accuracy/avg"]

            # If validation itself is non-finite (e.g. because every batch this
            # epoch was skipped and the model never actually updated, or -- pre
            # v2 -- because it was already corrupted), don't feed NaN into the
            # LR scheduler; ReduceLROnPlateau's internal comparisons with NaN
            # are silently always False, which would otherwise mean it can
            # never fire again for the rest of training.
            if not (val_fpr == val_fpr):  # NaN check without needing torch/np here
                self.logger.error(
                    f"  Epoch {epoch}: validation FPR is NaN. This means every "
                    f"training batch this epoch was skipped as non-finite, or "
                    f"the model state itself is already corrupted. Restoring "
                    f"the last known-good model/optimizer state and continuing."
                )
                self._restore_last_good_state()
            else:
                self.scheduler.step(val_fpr)

            self._log_epoch(epoch, train_metrics, val_metrics, epoch_time)
            self._write_tensorboard(epoch, train_metrics, val_metrics)

            improved = (
                val_fpr == val_fpr  # not NaN
                and val_fpr < self.best_val_fpr
                and val_acc >= tcfg.min_accuracy_threshold
            )
            if improved:
                self.best_val_fpr = val_fpr
                self.best_val_acc = val_acc
                self.best_epoch   = epoch
                self.no_improve   = 0
                self._save_checkpoint(epoch, tag="best")
                self.logger.info(
                    f"  ✓ New best model: FPR={val_fpr:.4f}, Acc={val_acc:.4f}"
                )
            else:
                self.no_improve += 1

            if epoch % 10 == 0:
                self._save_checkpoint(epoch, tag=f"epoch{epoch}")

            if self.no_improve >= tcfg.early_stop_patience:
                self.logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {tcfg.early_stop_patience} epochs)"
                )
                break

            self.history.append({
                "epoch": epoch,
                "train": train_metrics,
                "val":   val_metrics,
            })

        self.logger.info(
            f"Training complete. Best epoch: {self.best_epoch} | "
            f"Best val FPR: {self.best_val_fpr:.4f} | "
            f"Best val Acc: {self.best_val_acc:.4f}"
        )
        if self.nan_incidents:
            self.logger.warning(
                f"{len(self.nan_incidents)} non-finite batch(es) were detected "
                f"and skipped during training (see {self.cfg.training.log_dir}/"
                f"{self.cfg.training.run_name}_nan_incidents.json for the exact "
                f"Taskset_IDs). The model was NOT corrupted by them, but you "
                f"should inspect those rows in your source data -- a real bug "
                f"almost always means something in that row is malformed."
            )
            self._save_nan_incidents()
        if self.recovery_events:
            self.logger.warning(
                f"{len(self.recovery_events)} training-instability rollback(s) "
                f"occurred (model+optimizer restored to last good state, LR "
                f"reduced each time -- see {self.cfg.training.log_dir}/"
                f"{self.cfg.training.run_name}_recovery_events.json). If this "
                f"happens repeatedly across a run, treat it as a signal to "
                f"lower --encoder_lr / --head_lr, tighten --grad_clip, or "
                f"reduce --fp_lambda / --hi_mu up front rather than relying on "
                f"the automatic recovery every time."
            )
            self._save_recovery_events()
        self._save_history()
        self.writer.close()

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch. Returns aggregated training metrics."""
        tcfg = self.cfg.training
        self.model.train()

        total_loss_sum = 0.0
        bce_sum        = 0.0
        fp_sum         = 0.0
        hi_fp_sum      = 0.0
        n_batches      = 0
        n_skipped      = 0

        all_probs  = []
        all_labels = []
        all_n      = []

        for batch_idx, batch in enumerate(self.train_loader):
            batch = self._to_device(batch)

            # ── NEW: guard 1 — reject non-finite INPUTS before they ever reach
            # the model. This is the cheapest possible check and, if it fires,
            # tells you immediately that the problem is in the data, not the
            # model or the optimizer.
            bad_input_tensor = _first_nonfinite_tensor_name(batch)
            if bad_input_tensor is not None:
                self._handle_bad_batch(
                    epoch, batch_idx, batch,
                    reason=f"non-finite values in batch['{bad_input_tensor}'] "
                          f"(input data, before the model ever ran)",
                )
                n_skipped += 1
                self._register_nonfinite_streak(epoch, batch_idx)
                continue

            # ── Forward ────────────────────────────────────────────────────
            out        = self.model(batch)
            logits     = out["logits"]
            raw_logits = out["raw_logits"]

            prob_tensor = torch.stack(
                [logits[name] for name in self.cfg.model.scheduler_names], dim=1
            )

            # ── NEW: guard 2 — reject a non-finite FORWARD PASS before it
            # ever reaches the loss or backward. If guard 1 passed but this
            # fires, the problem is numerical instability inside the model
            # for this specific (finite!) input, not a malformed CSV row.
            if not torch.isfinite(prob_tensor).all():
                self._handle_bad_batch(
                    epoch, batch_idx, batch,
                    reason="non-finite model output (forward pass produced "
                          "NaN/Inf from finite inputs -- likely a numerical "
                          "instability inside the model, not the data)",
                )
                n_skipped += 1
                self._register_nonfinite_streak(epoch, batch_idx)
                continue

            # ── Loss ───────────────────────────────────────────────────────
            loss_dict = self.criterion(
                logits     = logits,
                labels     = batch["labels"],
                features   = batch["features"],
                mask       = batch["mask"],
                raw_logits = raw_logits,
            )
            loss = loss_dict["total"]

            # ── NEW: guard 3 — reject a non-finite LOSS before backward().
            # This is the guard that actually prevents the epoch-7 collapse:
            # once a NaN reaches loss.backward(), gradient-norm clipping
            # multiplies EVERY parameter's gradient by the same NaN scale
            # factor, corrupting the entire model in a single optimizer.step().
            # Skipping here means that never happens.
            if not torch.isfinite(loss):
                self._handle_bad_batch(
                    epoch, batch_idx, batch,
                    reason="non-finite loss (forward pass was finite, but the "
                          "loss computation produced NaN/Inf)",
                )
                n_skipped += 1
                self._register_nonfinite_streak(epoch, batch_idx)
                continue

            # ── Backward ───────────────────────────────────────────────────
            self.optimizer.zero_grad()
            loss.backward()

            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.training.grad_clip
            )

            # ── NEW: guard 4 — reject a non-finite GRADIENT NORM before
            # optimizer.step(). Covers the rare case where forward+loss are
            # both finite but backward() itself produces a NaN/Inf gradient
            # (e.g. from a 0/0 at a clamp boundary during backprop).
            if not torch.isfinite(grad_norm):
                self._handle_bad_batch(
                    epoch, batch_idx, batch,
                    reason=f"non-finite gradient norm ({grad_norm.item()}) after "
                          f"backward() -- forward pass and loss were finite, so "
                          f"this originated inside backward()",
                )
                self.optimizer.zero_grad(set_to_none=True)
                n_skipped += 1
                self._register_nonfinite_streak(epoch, batch_idx)
                continue

            self.optimizer.step()

            # A fully clean end-to-end step: the instability streak resets.
            # Only CONSECUTIVE non-finite batches indicate the model itself
            # has drifted into an unstable region; an isolated bad batch
            # sandwiched between clean ones is normal (almost always one
            # malformed data row) and should not trigger a rollback.
            self.consecutive_nonfinite = 0

            # This batch was fully clean end-to-end: periodically remember it
            # as the last known-good state so we can roll back if something
            # later in the epoch still manages to corrupt the model despite
            # the guards above (defense in depth, not expected to trigger
            # often). Snapshotting the full state_dict is not free, so this
            # is throttled to every log_interval steps rather than every step.
            if batch_idx == 0 or (batch_idx + 1) % tcfg.log_interval == 0:
                self._snapshot_good_state()

            # ── Accumulate ─────────────────────────────────────────────────
            total_loss_sum += loss.item()
            bce_sum        += loss_dict["bce"].item()
            fp_sum         += loss_dict["fp_penalty"].item()
            hi_fp_sum      += loss_dict["hi_fp_penalty"].item()
            n_batches      += 1

            all_probs.append(prob_tensor.detach().cpu())
            all_labels.append(batch["labels"].cpu())
            all_n.extend(batch["n_tasks"])

            if (batch_idx + 1) % tcfg.log_interval == 0:
                self.logger.debug(
                    f"  Epoch {epoch} | Batch {batch_idx+1}/{len(self.train_loader)} | "
                    f"Loss={loss.item():.4f} | BCE={loss_dict['bce'].item():.4f} | "
                    f"FP={loss_dict['fp_penalty'].item():.4f} | "
                    f"HI_FP={loss_dict['hi_fp_penalty'].item():.4f}"
                )

        if n_skipped:
            self.logger.warning(
                f"  Epoch {epoch}: skipped {n_skipped}/{len(self.train_loader)} "
                f"non-finite batch(es); {n_batches} clean batches used for the "
                f"optimizer this epoch."
            )

        # ── Epoch-level training metrics ────────────────────────────────────
        if n_batches == 0:
            # Every batch this epoch was non-finite. Return NaN-flagged metrics
            # rather than crashing on an empty accumulate_predictions call, so
            # train() can detect this via the NaN check on val_fpr next.
            self.logger.error(
                f"  Epoch {epoch}: ALL batches were skipped as non-finite. "
                f"No optimizer step occurred this epoch."
            )
            return {
                "loss/total": float("nan"), "loss/bce": float("nan"),
                "loss/fp": float("nan"), "loss/hi_fp": float("nan"),
                "fpr/avg": float("nan"), "precision/avg": float("nan"),
                "recall/avg": float("nan"), "accuracy/avg": float("nan"),
                "auroc/avg": float("nan"),
            }

        probs_cat, labels_cat, n_cat = accumulate_predictions(all_probs, all_labels, [all_n])
        train_metrics = compute_metrics(
            probs_cat, labels_cat,
            scheduler_names=self.cfg.model.scheduler_names,
        )
        train_metrics["loss/total"] = total_loss_sum / max(n_batches, 1)
        train_metrics["loss/bce"]   = bce_sum        / max(n_batches, 1)
        train_metrics["loss/fp"]    = fp_sum          / max(n_batches, 1)
        train_metrics["loss/hi_fp"] = hi_fp_sum       / max(n_batches, 1)

        return train_metrics

    # ── NEW: NaN handling helpers ────────────────────────────────────────────

    def _handle_bad_batch(self, epoch: int, batch_idx: int, batch: Dict, reason: str):
        """
        Log full diagnostic info for a non-finite batch (including the exact
        Taskset_IDs so the offending CSV row(s) can be found and fixed) and
        record the incident. Does NOT touch the optimizer -- callers are
        responsible for skipping the step.
        """
        taskset_ids = batch.get("taskset_ids", None)
        n_tasks     = batch.get("n_tasks", None)

        self.logger.error(
            f"  [NON-FINITE BATCH] epoch={epoch} batch={batch_idx} | {reason}\n"
            f"    Taskset_IDs in this batch: {taskset_ids}\n"
            f"    n_tasks in this batch    : {n_tasks}\n"
            f"    Skipping this batch (no optimizer step). Training continues."
        )
        self.nan_incidents.append({
            "epoch": epoch,
            "batch_idx": batch_idx,
            "reason": reason,
            "taskset_ids": taskset_ids,
            "n_tasks": n_tasks,
        })

    def _snapshot_good_state(self):
        """
        Cheap CPU-side snapshot of model+optimizer state after a fully clean
        step. Only kept in memory (not written to disk every step -- that
        would be far too slow); used as a rollback point both for the rare
        "guards missed something" case AND -- the common case in practice --
        for `_recover_from_instability` below.

        The OPTIMIZER state is snapshotted too, not just the model weights.
        This matters: AdamW keeps per-parameter running mean/variance
        estimates (exp_avg / exp_avg_sq). If only the model weights are
        rolled back but the optimizer's momentum state is left as-is, the
        very same running estimates that pushed a weight into the unstable
        region in the first place are still in effect, and training can walk
        straight back into the same failure within a handful of steps.
        Restoring both gives the optimizer a genuinely clean slate.
        """
        self._last_good_state = {
            "model": {k: v.detach().clone() for k, v in self.model.state_dict().items()},
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
        }

    def _restore_last_good_state(self):
        if self._last_good_state is None:
            self.logger.error(
                "  No known-good state available to restore -- model may be "
                "corrupted. Recommend stopping and inspecting the NaN "
                "incident log."
            )
            return
        self.model.load_state_dict(self._last_good_state["model"])
        self.optimizer.load_state_dict(self._last_good_state["optimizer"])
        self.logger.info("  Restored model AND optimizer to last known-good "
                         "(fully finite) state.")

    def _register_nonfinite_streak(self, epoch: int, batch_idx: int):
        """
        Called every time a batch is skipped for being non-finite (any of
        guards 1-4). Tracks how many such batches have happened IN A ROW.

        A single skipped batch is normal and not acted on further here --
        almost always one malformed data row, already logged and handled by
        _handle_bad_batch. But `nonfinite_recovery_patience` (default 5)
        CONSECUTIVE non-finite batches, regardless of which Taskset_IDs they
        contain, is a categorically different signal: the model's weights
        have drifted into a numerically unstable region where backward()
        overflows for essentially any input. Left alone, every subsequent
        batch for the rest of training would hit the same wall and the run
        would silently do nothing for all remaining epochs while still
        "completing" without crashing. This method detects that condition
        and triggers a rollback + LR reduction instead of stalling forever.
        """
        self.consecutive_nonfinite += 1
        if self.consecutive_nonfinite >= self.nonfinite_recovery_patience:
            self._recover_from_instability(epoch, batch_idx)

    def _recover_from_instability(self, epoch: int, batch_idx: int):
        streak = self.consecutive_nonfinite
        old_lrs = [g["lr"] for g in self.optimizer.param_groups]

        self.logger.error(
            f"\n{'='*78}\n"
            f"TRAINING INSTABILITY DETECTED\n"
            f"{'='*78}\n"
            f"  {streak} consecutive non-finite batches at epoch={epoch}, "
            f"batch={batch_idx}.\n"
            f"  This many failures in a row (independent of which Taskset_IDs\n"
            f"  are involved) means the MODEL's weights have drifted into a\n"
            f"  numerically unstable region, not that a handful of data rows\n"
            f"  are malformed. Rolling back to the last known-good checkpoint\n"
            f"  and reducing the learning rate by "
            f"{self.lr_reduction_factor_on_instability:.2f}x before continuing.\n"
            f"{'='*78}\n"
        )

        self._restore_last_good_state()

        # Restoring the optimizer's state_dict also resets 'lr' in every
        # param_group back to whatever was in effect at snapshot time. If
        # this is the SECOND (or later) recovery to fire without a clean
        # snapshot in between, that would silently UNDO the previous
        # reduction -- the LR could bounce back up instead of decreasing
        # monotonically. Learning rate is a schedule we're deliberately
        # adjusting, not part of the "did this step corrupt the model" state
        # we want rolled back, so it's explicitly reapplied at its
        # pre-restore value here before the new reduction is applied.
        for g, lr in zip(self.optimizer.param_groups, old_lrs):
            g["lr"] = lr

        for g in self.optimizer.param_groups:
            g["lr"] = max(
                g["lr"] * self.lr_reduction_factor_on_instability,
                self.cfg.training.lr_min,
            )
        new_lrs = [g["lr"] for g in self.optimizer.param_groups]
        self.logger.info(f"  LR changed: {old_lrs} -> {new_lrs}")

        self.recovery_events.append({
            "epoch": epoch,
            "batch_idx": batch_idx,
            "consecutive_nonfinite": streak,
            "lr_before": old_lrs,
            "lr_after": new_lrs,
        })

        self.consecutive_nonfinite = 0

    def _save_nan_incidents(self):
        tcfg = self.cfg.training
        path = os.path.join(tcfg.log_dir, f"{tcfg.run_name}_nan_incidents.json")
        with open(path, "w") as f:
            json.dump(self.nan_incidents, f, indent=2)
        self.logger.info(f"NaN/Inf incident log saved to {path}")

    def _save_recovery_events(self):
        tcfg = self.cfg.training
        path = os.path.join(tcfg.log_dir, f"{tcfg.run_name}_recovery_events.json")
        with open(path, "w") as f:
            json.dump(self.recovery_events, f, indent=2)
        self.logger.info(f"Instability recovery event log saved to {path}")

    # ── Evaluation ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader, split: str = "val") -> Dict[str, float]:
        """Evaluate on a DataLoader. Returns metrics dict."""
        self.model.eval()

        all_probs  = []
        all_labels = []
        all_n_lists= []
        total_loss = 0.0
        n_batches  = 0

        for batch in loader:
            batch = self._to_device(batch)

            if _first_nonfinite_tensor_name(batch) is not None:
                # Non-finite eval inputs would silently poison metrics (e.g.
                # AUROC) rather than the model; skip and log instead.
                self.logger.error(
                    f"  [{split}] Skipping non-finite batch: "
                    f"Taskset_IDs={batch.get('taskset_ids')}"
                )
                continue

            out        = self.model(batch)
            logits     = out["logits"]
            raw_logits = out["raw_logits"]

            prob_tensor = torch.stack(
                [logits[name] for name in self.cfg.model.scheduler_names], dim=1
            )
            if not torch.isfinite(prob_tensor).all():
                self.logger.error(
                    f"  [{split}] Skipping batch with non-finite model output: "
                    f"Taskset_IDs={batch.get('taskset_ids')}"
                )
                continue

            loss_dict = self.criterion(
                logits     = logits,
                labels     = batch["labels"],
                features   = batch["features"],
                mask       = batch["mask"],
                raw_logits = raw_logits,
            )
            total_loss += loss_dict["total"].item()
            n_batches  += 1

            all_probs.append(prob_tensor.cpu())
            all_labels.append(batch["labels"].cpu())
            all_n_lists.append(batch["n_tasks"])

        if n_batches == 0:
            return {
                "loss/total": float("nan"), "fpr/avg": float("nan"),
                "precision/avg": float("nan"), "recall/avg": float("nan"),
                "accuracy/avg": float("nan"), "auroc/avg": float("nan"),
            }

        probs_cat, labels_cat, n_cat = accumulate_predictions(
            all_probs, all_labels, all_n_lists
        )
        metrics = compute_metrics(
            probs_cat, labels_cat,
            scheduler_names=self.cfg.model.scheduler_names,
        )
        metrics["loss/total"] = total_loss / max(n_batches, 1)

        return metrics

    @torch.no_grad()
    def evaluate_generalization(self, checkpoint: str = "best"):
        """
        Load the best checkpoint and evaluate on all generalization splits.
        Also evaluates on in-distribution test split.
        Saves per-n breakdown to JSON.
        """
        self._load_checkpoint(checkpoint)
        self.model.eval()
        tcfg = self.cfg.training

        all_results = {}

        test_metrics = self._evaluate(self.test_loader, split="test")
        all_results["test_in_dist"] = test_metrics
        self.logger.info(
            f"[test_in_dist] FPR={test_metrics['fpr/avg']:.4f} | "
            f"Prec={test_metrics['precision/avg']:.4f} | "
            f"Rec={test_metrics['recall/avg']:.4f} | "
            f"Acc={test_metrics['accuracy/avg']:.4f}"
        )

        for name, gen_item in zip(tcfg.gen_names, self.gen_items):
            # gen_item is a LazyGenSplit handle (memory-safe default) or,
            # if the trainer was built with eager_gen=True, an already-built
            # DataLoader/None (kept for backward compatibility only).
            if hasattr(gen_item, "load"):
                loader = gen_item.load()
            else:
                loader = gen_item

            if loader is None:
                self.logger.warning(f"  Generalization split '{name}' not found, skipping.")
                continue

            all_probs, all_labels, all_n_lists = [], [], []
            for batch in loader:
                batch = self._to_device(batch)
                if _first_nonfinite_tensor_name(batch) is not None:
                    self.logger.error(
                        f"  [{name}] Skipping non-finite batch: "
                        f"Taskset_IDs={batch.get('taskset_ids')}"
                    )
                    continue
                out   = self.model(batch)
                prob_tensor = torch.stack(
                    [out["logits"][s] for s in self.cfg.model.scheduler_names], dim=1
                )
                if not torch.isfinite(prob_tensor).all():
                    self.logger.error(
                        f"  [{name}] Skipping batch with non-finite model output: "
                        f"Taskset_IDs={batch.get('taskset_ids')}"
                    )
                    continue
                all_probs.append(prob_tensor.cpu())
                all_labels.append(batch["labels"].cpu())
                all_n_lists.append(batch["n_tasks"])

            if not all_probs:
                self.logger.warning(f"  [{name}] No valid batches; skipping split.")
                continue

            probs_cat, labels_cat, n_cat = accumulate_predictions(
                all_probs, all_labels, all_n_lists
            )

            split_metrics = compute_metrics(
                probs_cat, labels_cat,
                scheduler_names=self.cfg.model.scheduler_names,
            )

            per_n = compute_metrics_by_n(
                probs_cat, labels_cat, n_cat,
                scheduler_names=self.cfg.model.scheduler_names,
            )

            all_results[name] = {
                "overall": split_metrics,
                "per_n":   {str(k): v for k, v in per_n.items()},
            }

            self.logger.info(
                f"[{name}] FPR={split_metrics['fpr/avg']:.4f} | "
                f"Prec={split_metrics['precision/avg']:.4f} | "
                f"Rec={split_metrics['recall/avg']:.4f} | "
                f"Acc={split_metrics['accuracy/avg']:.4f}"
            )

            for n_val, m in sorted(per_n.items()):
                self.logger.info(
                    f"  n={n_val:4d}: FPR={m['fpr/avg']:.4f} | "
                    f"Prec={m['precision/avg']:.4f} | Rec={m['recall/avg']:.4f}"
                )

            # ── NEW: free this split's dataset/loader/tensors before moving
            # to the next one. With 5 generalization splits potentially each
            # holding hundreds of thousands of rows, keeping all of them
            # resident simultaneously (the old eager behavior) is exactly the
            # kind of unnecessary peak-memory cost that caused the OOM kill.
            del loader, all_probs, all_labels, all_n_lists, probs_cat, labels_cat
            import gc as _gc
            _gc.collect()

        out_path = os.path.join(tcfg.log_dir, f"{tcfg.run_name}_gen_results.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        self.logger.info(f"Generalization results saved to {out_path}")

        return all_results

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _to_device(self, batch: Dict) -> Dict:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _save_checkpoint(self, epoch: int, tag: str = "best"):
        tcfg = self.cfg.training
        path = os.path.join(
            tcfg.checkpoint_dir,
            f"{tcfg.run_name}_{tag}.pt",
        )
        torch.save(
            {
                "epoch":           epoch,
                "model_state":     self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "best_val_fpr":    self.best_val_fpr,
                "best_val_acc":    self.best_val_acc,
                "norm_mean":       self.norm_stats["mean"],
                "norm_std":        self.norm_stats["std"],
            },
            path,
        )
        self.logger.debug(f"Checkpoint saved: {path}")

    def _load_checkpoint(self, tag: str = "best"):
        tcfg = self.cfg.training
        path = os.path.join(tcfg.checkpoint_dir, f"{tcfg.run_name}_{tag}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state"])
        self.logger.info(f"Loaded checkpoint from {path} (epoch {ckpt['epoch']})")

    def _log_epoch(
        self,
        epoch: int,
        train: Dict,
        val: Dict,
        elapsed: float,
    ):
        snames = self.cfg.model.scheduler_names
        lr_enc = self.optimizer.param_groups[0]["lr"]
        lr_hd  = self.optimizer.param_groups[1]["lr"]

        header = (
            f"Epoch {epoch:3d}/{self.cfg.training.num_epochs} | "
            f"t={elapsed:.1f}s | LR_enc={lr_enc:.2e} LR_head={lr_hd:.2e}"
        )
        train_line = (
            f"  TRAIN | Loss={train['loss/total']:.4f} | "
            f"BCE={train['loss/bce']:.4f} | FP={train['loss/fp']:.4f} | "
            f"FPR={train['fpr/avg']:.4f} | Prec={train['precision/avg']:.4f} | "
            f"Rec={train['recall/avg']:.4f} | Acc={train['accuracy/avg']:.4f}"
        )
        val_line = (
            f"  VAL   | Loss={val['loss/total']:.4f} | "
            f"FPR={val['fpr/avg']:.4f} | Prec={val['precision/avg']:.4f} | "
            f"Rec={val['recall/avg']:.4f} | Acc={val['accuracy/avg']:.4f} | "
            f"AUROC={val['auroc/avg']:.4f}"
        )

        per_sched = "  " + " | ".join(
            f"{n}: FPR={val[f'fpr/{n}']:.4f}" for n in snames
        )

        self.logger.info(header)
        self.logger.info(train_line)
        self.logger.info(val_line)
        self.logger.info(per_sched)

    def _write_tensorboard(self, epoch: int, train: Dict, val: Dict):
        for key, val_val in val.items():
            if isinstance(val_val, float):
                self.writer.add_scalar(f"val/{key}", val_val, epoch)
        for key, trn_val in train.items():
            if isinstance(trn_val, float):
                self.writer.add_scalar(f"train/{key}", trn_val, epoch)

        self.writer.add_scalar("lr/encoder", self.optimizer.param_groups[0]["lr"], epoch)
        self.writer.add_scalar("lr/heads",   self.optimizer.param_groups[1]["lr"], epoch)

    def _save_history(self):
        tcfg = self.cfg.training
        path = os.path.join(tcfg.log_dir, f"{tcfg.run_name}_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        self.logger.info(f"Training history saved to {path}")