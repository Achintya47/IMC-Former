"""
train.py — IMC-Former training entry point
==========================================
Run:
    python train.py                            # use all defaults
    python train.py --batch_size 128           # override one field
    python train.py --run_name my_exp          # named experiment
    python train.py --device cpu               # force CPU
    python train.py --data_dir /custom/data    # custom data directory
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from configs.config import IMCFormerConfig, ModelConfig, LossConfig, TrainingConfig
from training.trainer import IMCFormerTrainer


def parse_args():
    p = argparse.ArgumentParser(description="Train IMC-Former")

    # ── Training overrides ──
    p.add_argument("--data_dir",    default="data",           help="Directory with split CSVs")
    p.add_argument("--log_dir",     default="logs",           help="Log output directory")
    p.add_argument("--ckpt_dir",    default="checkpoints",    help="Checkpoint directory")
    p.add_argument("--fig_dir",     default="figures",        help="Figures directory")
    p.add_argument("--run_name",    default="imc_former_base",help="Experiment name (used in filenames)")
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--num_epochs",  type=int,   default=80)
    p.add_argument("--encoder_lr",  type=float, default=3e-4)
    p.add_argument("--head_lr",     type=float, default=1e-3)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--num_workers", type=int,   default=4)

    # ── Loss overrides ──
    p.add_argument("--fp_lambda",   type=float, default=2.0,  help="FP penalty λ")
    p.add_argument("--hi_mu",       type=float, default=3.0,  help="HI-FP penalty μ")
    p.add_argument("--hi_thresh",   type=float, default=0.7,  help="HI-critical U_HI threshold")

    # ── Model overrides ──
    p.add_argument("--d_model",     type=int,   default=256)
    p.add_argument("--n_heads",     type=int,   default=8)
    p.add_argument("--n_layers",    type=int,   default=3,    help="Transformer layers per stream")
    p.add_argument("--chunk_size",  type=int,   default=8,    help="Chunk size for hierarchical pooling")

    # ── Evaluation only ──
    p.add_argument("--eval_only",   action="store_true",
                   help="Skip training; load best checkpoint and run generalization eval")

    return p.parse_args()


def build_config(args) -> IMCFormerConfig:
    model = ModelConfig(
        d_model            = args.d_model,
        n_heads            = args.n_heads,
        n_transformer_layers = args.n_layers,
        chunk_size         = args.chunk_size,
        fused_dim          = args.d_model * 2 + 64,   # 2d + context_hidden(64)
    )
    loss = LossConfig(
        fp_penalty_lambda  = args.fp_lambda,
        hi_fp_penalty_mu   = args.hi_mu,
        hi_critical_threshold = args.hi_thresh,
    )
    training = TrainingConfig(
        data_dir           = args.data_dir,
        log_dir            = args.log_dir,
        checkpoint_dir     = args.ckpt_dir,
        figure_dir         = args.fig_dir,
        run_name           = args.run_name,
        batch_size         = args.batch_size,
        num_epochs         = args.num_epochs,
        encoder_lr         = args.encoder_lr,
        head_lr            = args.head_lr,
        weight_decay       = args.weight_decay,
        seed               = args.seed,
        device             = args.device,
        num_workers        = args.num_workers,
    )
    return IMCFormerConfig(model=model, loss=loss, training=training).validate()


def main():
    args   = parse_args()
    cfg    = build_config(args)

    trainer = IMCFormerTrainer(cfg)

    if args.eval_only:
        print("Eval-only mode: loading best checkpoint and running generalization eval.")
        results = trainer.evaluate_generalization(checkpoint="best")
    else:
        trainer.train()
        results = trainer.evaluate_generalization(checkpoint="best")

    print("\nDone.")


if __name__ == "__main__":
    main()
