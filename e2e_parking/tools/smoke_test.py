"""检查注册表、构建、训练一步、推理一次。python tools/smoke_test.py [configs/smoke_synthetic.py] [--device cpu]"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import parking_e2e  # noqa: E402,F401
from parking_e2e import DATASETS, HEADS, MODELS, Config  # noqa: E402
from parking_e2e.data import collate_parking, move_to_device  # noqa: E402
from parking_e2e.geometry import KannalaBrandt, Pinhole  # noqa: E402


def check_cameras():
    torch.manual_seed(0)
    z = torch.rand(1000) * 8 + 2
    pts = torch.stack([(torch.rand(1000) - 0.5) * 1.2 * z, (torch.rand(1000) - 0.5) * 1.2 * z, z], 1)
    intr = torch.tensor([300.0, 300.0, 256.0, 160.0]).expand(1000, 4)
    for cam, dist in ((Pinhole(), torch.tensor([0.05, -0.01, 0.001, 0.001, 0.0]).expand(1000, 5)),
                      (KannalaBrandt(), torch.tensor([0.05, -0.01, 0.002, -0.0005]).expand(1000, 4))):
        uv, depth, valid = cam.project(pts, intr, dist)
        rays, rvalid = cam.unproject(uv, intr, dist)
        rec = rays * depth[:, None]
        err = (rec - pts)[valid & rvalid].norm(dim=1).max().item()
        print(f"{type(cam).__name__}: project->unproject max error {err:.2e} m")
        assert err < 1e-2, "camera model round trip failed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default=os.path.join(os.path.dirname(__file__), "..", "configs", "smoke_synthetic.py"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print("registered heads:", sorted(HEADS.module_dict))
    check_cameras()

    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model).to(args.device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model built: {n_params:.1f}M params, heads={list(model.heads)}")

    ds = DATASETS.build(cfg.train_dataset)
    batch = move_to_device(collate_parking([ds[i] for i in range(cfg.data_loader.batch_size)]), args.device)

    model.train()
    model.set_loss_weights(dict(traj=1.0, slot=1.0))
    losses = model(batch, mode="loss")
    losses["loss"].backward()
    for k, v in sorted(losses.items()):
        print(f"  {k:32s} {float(v):.4f}")
    assert torch.isfinite(losses["loss"]), "loss is not finite"
    no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    print(f"params without grad: {len(no_grad)}", no_grad[:10])

    model.eval()
    out = model(batch, mode="predict")
    for k, v in out.items():
        desc = tuple(v.shape) if torch.is_tensor(v) else f"list[{len(v)}]"
        print(f"  pred {k:20s} {desc}")
    print("smoke test passed")


if __name__ == "__main__":
    main()
