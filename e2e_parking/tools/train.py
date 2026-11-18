import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import parking_e2e  # noqa: E402,F401  触发全部注册
from parking_e2e import Config  # noqa: E402
from parking_e2e.engine import Trainer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--work-dir", default=None)
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    if args.work_dir:
        cfg.train_cfg.work_dir = args.work_dir
    trainer = Trainer(cfg, device=args.device)
    if args.resume:
        trainer.resume(args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
