# 冒烟测试：小网格、小图、随机数据，CPU 也能跑通一次前向 + 反向 + 推理。
_base_ = ["parking_e2e_r18.py"]

grid = dict(xbound=(-12.0, 12.0, 0.4), ybound=(-12.0, 12.0, 0.4), zbound=(-0.4, 2.0, 0.4))
img_sizes = dict(pinhole=(128, 224), fisheye=(160, 256))

model = dict(
    grid=grid,
    img_backbone=dict(weights=None),
    bev_temporal=dict(num_history=1),
    voxel_temporal=dict(num_history=1),
    heads=dict(
        bev_det=dict(num_queries=50),
        slot=dict(num_queries=8),
        traj=dict(num_samples=4, sample_steps=4),
    ),
    deploy_heads=["pv_depth", "pv_seg", "pv_det", "bev_seg", "bev_det", "occ", "occ_flow", "sdf", "slot", "traj"],
)

train_dataset = dict(
    _delete_=True, type="SyntheticParkingDataset", grid=grid, length=4, img_sizes=img_sizes,
    bev_seg_upsample=4, num_occ_pts=2048, num_sdf_pts=2048, num_flow_pts=512, num_history=1,
)
data_loader = dict(batch_size=2, num_workers=0, shuffle=False)
train_cfg = dict(
    epochs=2, log_interval=1, amp=False, work_dir="work_dirs/smoke",
    stages=[dict(start_epoch=0, loss_weights=dict(traj=0.0, slot=0.0)),
            dict(start_epoch=1, loss_weights=dict(traj=1.0, slot=1.0))],
)
scheduler = dict(warmup_iters=2)
