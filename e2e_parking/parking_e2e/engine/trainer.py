"""单机训练循环。两段训练：第一段只开辅助头（规划权重为 0），第二段加上规划损失。

train_cfg = dict(
    epochs=24, log_interval=20, grad_clip=35.0, amp=True, work_dir="work_dirs/xxx",
    stages=[dict(start_epoch=0, loss_weights=dict(traj=0.0, slot=0.0)),
            dict(start_epoch=12, loss_weights=dict(traj=1.0, slot=1.0))],
)
"""
import os
import time
from collections import defaultdict
from typing import Dict

import torch
from torch.utils.data import DataLoader

from ..data import collate_parking, move_to_device
from ..registry import DATASETS, MODELS, SCHEDULERS
from .optim import build_optimizer


class Trainer:
    def __init__(self, cfg: Dict, device: str = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model = MODELS.build(cfg["model"]).to(self.device)
        dl = cfg.get("data_loader", {})
        self.dataset = DATASETS.build(cfg["train_dataset"])
        self.loader = DataLoader(self.dataset, batch_size=dl.get("batch_size", 1), shuffle=dl.get("shuffle", True),
                                 num_workers=dl.get("num_workers", 4), collate_fn=collate_parking,
                                 pin_memory=self.device.type == "cuda", drop_last=True,
                                 persistent_workers=dl.get("num_workers", 4) > 0)
        tc = cfg["train_cfg"]
        self.epochs = tc["epochs"]
        self.optimizer = build_optimizer(self.model, cfg["optimizer"])
        sched_cfg = dict(cfg.get("scheduler", dict(type="WarmupCosine")))
        sched_cfg.setdefault("total_iters", self.epochs * len(self.loader))
        self.scheduler = SCHEDULERS.build(sched_cfg, optimizer=self.optimizer)
        self.amp = bool(tc.get("amp", False)) and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.grad_clip = tc.get("grad_clip")
        self.log_interval = tc.get("log_interval", 20)
        self.work_dir = tc.get("work_dir", "work_dirs/parking_e2e")
        self.stages = sorted(tc.get("stages", []), key=lambda s: s["start_epoch"])
        self.start_epoch = 0
        os.makedirs(self.work_dir, exist_ok=True)

    def _apply_stage(self, epoch: int):
        for st in self.stages:
            if st["start_epoch"] == epoch:
                self.model.set_loss_weights(st.get("loss_weights", {}))
                print(f"[stage] epoch {epoch}: loss weights -> {self.model.loss_weights}")

    def resume(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.start_epoch = ckpt["epoch"] + 1
        for st in self.stages:
            if st["start_epoch"] <= ckpt["epoch"]:
                self.model.set_loss_weights(st.get("loss_weights", {}))

    def train_step(self, batch) -> Dict[str, float]:
        batch = move_to_device(batch, self.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp):
            losses = self.model(batch, mode="loss")
        loss = losses["loss"]
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        if self.grad_clip:
            self.scaler.unscale_(self.optimizer)
            gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            losses["grad_norm"] = gnorm
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        return {k: float(v) for k, v in losses.items() if torch.is_tensor(v)}

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            self._apply_stage(epoch)
            self.model.train()
            meter = defaultdict(float)
            n, t0 = 0, time.time()
            for it, batch in enumerate(self.loader):
                logs = self.train_step(batch)
                for k, v in logs.items():
                    meter[k] += v
                n += 1
                if (it + 1) % self.log_interval == 0 or it + 1 == len(self.loader):
                    msg = " ".join(f"{k}={v / n:.4f}" for k, v in sorted(meter.items()))
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(f"[epoch {epoch} iter {it + 1}/{len(self.loader)}] lr={lr:.2e} "
                          f"{(time.time() - t0) / n:.2f}s/it {msg}", flush=True)
                    meter.clear()
                    n, t0 = 0, time.time()
            torch.save(dict(model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                            scheduler=self.scheduler.state_dict(), epoch=epoch),
                       os.path.join(self.work_dir, f"epoch_{epoch}.pth"))
