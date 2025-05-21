#!/usr/bin/env python
import argparse, subprocess, sys, textwrap

def parse():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Thin wrapper that forwards arguments to train_model.py.
            Keeps your SBATCH script clean and lets you add cluster-only
            options here later (e.g. wandb resume, fsync, etc.).
        """))
    p.add_argument("--data-dir", required=True)
    p.add_argument("--epochs",   type=int, default=25)
    p.add_argument("--vars",     nargs="+", default=["TMQ"])
    return p.parse_args()

if __name__ == "__main__":
    args = parse()

    cmd = ["python", "train_model.py",
           "--data-dir", args.data_dir,
           "--epochs",   str(args.epochs),
           "--vars",    *args.vars]

    print("⇢ Launching:", " ".join(cmd), flush=True)
    # propagate exit code (fails job if training crashes)
    sys.exit(subprocess.call(cmd))
