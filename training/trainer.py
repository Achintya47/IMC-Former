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
"""

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
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (full debug log)
    fh = logging.FileHandler(os.path.join(log_dir, f"{run_name}.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


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
        self.logger.info("Loading datasets ...")
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.gen_loaders,
            self.norm_stats,
        ) = build_dataloaders(cfg)
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
        # Compute positive class weights from training data to handle imbalance
        pos_weights = self.train_loader.dataset.get_scheduler_class_weights()
        self.logger.info(f"  BCE positive class weights: {pos_weights.tolist()}")

        self.criterion = IMCLoss(
            cfg=lcfg,
            scheduler_names=mcfg.scheduler_names,
            pos_weights=pos_weights,
        ).to(self.device)

        # ── Optimizer (differential LR) ──────────────────────────────────────────
        # Encoder + transformer: slower LR (stable representations)
        # Prediction heads: faster LR (policy-specific boundaries adapt quickly)
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
        # Monitors validation FPR (lower is better → mode="min")
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
            # Rebuild sampler for current curriculum phase
            self.train_loader = update_train_sampler(self.train_loader, epoch, self.cfg)

            epoch_start = time.time()
            train_metrics = self._train_epoch(epoch)
            val_metrics   = self._evaluate(self.val_loader, split="val")
            epoch_time    = time.time() - epoch_start

            # ── LR step on val FPR ──────────────────────────────────────────
            val_fpr = val_metrics["fpr/avg"]
            val_acc = val_metrics["accuracy/avg"]
            self.scheduler.step(val_fpr)

            # ── Logging ─────────────────────────────────────────────────────
            self._log_epoch(epoch, train_metrics, val_metrics, epoch_time)
            self._write_tensorboard(epoch, train_metrics, val_metrics)

            # ── Model selection ──────────────────────────────────────────────
            # Best model = lowest val FPR subject to accuracy >= threshold
            improved = (
                val_fpr < self.best_val_fpr
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

            # ── Periodic checkpoint ──────────────────────────────────────────
            if epoch % 10 == 0:
                self._save_checkpoint(epoch, tag=f"epoch{epoch}")

            # ── Early stopping ───────────────────────────────────────────────
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

        all_probs  = []
        all_labels = []
        all_n      = []

        for batch_idx, batch in enumerate(self.train_loader):
            batch = self._to_device(batch)

            # ── Forward ────────────────────────────────────────────────────
            out        = self.model(batch)
            logits     = out["logits"]      # probabilities
            raw_logits = out["raw_logits"]  # pre-sigmoid, for stable BCE

            # Stack probabilities into (B, S)
            prob_tensor = torch.stack(
                [logits[name] for name in self.cfg.model.scheduler_names], dim=1
            )

            # ── Loss ───────────────────────────────────────────────────────
            loss_dict = self.criterion(
                logits     = logits,
                labels     = batch["labels"],
                features   = batch["features"],
                mask       = batch["mask"],
                raw_logits = raw_logits,
            )
            loss = loss_dict["total"]

            # ── Backward ───────────────────────────────────────────────────
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.training.grad_clip
            )
            self.optimizer.step()

            # ── Accumulate ─────────────────────────────────────────────────
            total_loss_sum += loss.item()
            bce_sum        += loss_dict["bce"].item()
            fp_sum         += loss_dict["fp_penalty"].item()
            hi_fp_sum      += loss_dict["hi_fp_penalty"].item()
            n_batches      += 1

            all_probs.append(prob_tensor.detach().cpu())
            all_labels.append(batch["labels"].cpu())
            all_n.extend(batch["n_tasks"])

            # ── Batch-level log ────────────────────────────────────────────
            if (batch_idx + 1) % tcfg.log_interval == 0:
                self.logger.debug(
                    f"  Epoch {epoch} | Batch {batch_idx+1}/{len(self.train_loader)} | "
                    f"Loss={loss.item():.4f} | BCE={loss_dict['bce'].item():.4f} | "
                    f"FP={loss_dict['fp_penalty'].item():.4f} | "
                    f"HI_FP={loss_dict['hi_fp_penalty'].item():.4f}"
                )

        # ── Epoch-level training metrics ────────────────────────────────────
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
            out        = self.model(batch)
            logits     = out["logits"]
            raw_logits = out["raw_logits"]

            prob_tensor = torch.stack(
                [logits[name] for name in self.cfg.model.scheduler_names], dim=1
            )

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

        # ── In-distribution test ────────────────────────────────────────────
        test_metrics = self._evaluate(self.test_loader, split="test")
        all_results["test_in_dist"] = test_metrics
        self.logger.info(
            f"[test_in_dist] FPR={test_metrics['fpr/avg']:.4f} | "
            f"Prec={test_metrics['precision/avg']:.4f} | "
            f"Rec={test_metrics['recall/avg']:.4f} | "
            f"Acc={test_metrics['accuracy/avg']:.4f}"
        )

        # ── Generalization splits ────────────────────────────────────────────
        for name, loader in zip(tcfg.gen_names, self.gen_loaders):
            if loader is None:
                self.logger.warning(f"  Generalization split '{name}' not found, skipping.")
                continue

            all_probs, all_labels, all_n_lists = [], [], []
            for batch in loader:
                batch = self._to_device(batch)
                out   = self.model(batch)
                prob_tensor = torch.stack(
                    [out["logits"][s] for s in self.cfg.model.scheduler_names], dim=1
                )
                all_probs.append(prob_tensor.cpu())
                all_labels.append(batch["labels"].cpu())
                all_n_lists.append(batch["n_tasks"])

            probs_cat, labels_cat, n_cat = accumulate_predictions(
                all_probs, all_labels, all_n_lists
            )

            # Overall metrics for this split
            split_metrics = compute_metrics(
                probs_cat, labels_cat,
                scheduler_names=self.cfg.model.scheduler_names,
            )

            # Per-n breakdown
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

            # Per-n log
            for n_val, m in sorted(per_n.items()):
                self.logger.info(
                    f"  n={n_val:4d}: FPR={m['fpr/avg']:.4f} | "
                    f"Prec={m['precision/avg']:.4f} | Rec={m['recall/avg']:.4f}"
                )

        # Save results
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
        # Save only serialisation-safe objects (no dataclass instances).
        # norm_stats contains only tensors → safe.
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
        # weights_only=True is safe here: checkpoint contains only tensors + scalars.
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

        # Per-scheduler FPR breakdown
        per_sched = "  " + " | ".join(
            f"{n}: FPR={val[f'fpr/{n}']:.4f}" for n in snames
        )

        self.logger.info(header)
        self.logger.info(train_line)
        self.logger.info(val_line)
        self.logger.info(per_sched)

    def _write_tensorboard(self, epoch: int, train: Dict, val: Dict):
        """Log scalars to TensorBoard."""
        for key, val_val in val.items():
            if isinstance(val_val, float):
                self.writer.add_scalar(f"val/{key}", val_val, epoch)
        for key, trn_val in train.items():
            if isinstance(trn_val, float):
                self.writer.add_scalar(f"train/{key}", trn_val, epoch)

        # Log LR
        self.writer.add_scalar("lr/encoder", self.optimizer.param_groups[0]["lr"], epoch)
        self.writer.add_scalar("lr/heads",   self.optimizer.param_groups[1]["lr"], epoch)

    def _save_history(self):
        tcfg = self.cfg.training
        path = os.path.join(tcfg.log_dir, f"{tcfg.run_name}_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        self.logger.info(f"Training history saved to {path}")