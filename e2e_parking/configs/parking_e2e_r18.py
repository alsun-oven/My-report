# 端到端泊车：7 针孔 + 4 鱼眼 + Lidar + Radar + 导航
# 任务：轨迹、车位、重建(深度)、图像分割、图像检测、BEV 分割、BEV 检测、OCC、OCC Flow、SDF
#
# 近场范围：前后左右 12 m，高度 -0.4 ~ 2.2 m（相对局部地面）。
# 特征网格 0.2 m；BEV 分割上采样到 5 cm；GOD 头是隐式解码器，监督点来自 5 cm(5 m 内)/10 cm(5 m 外) 两级标签。

grid = dict(xbound=(-12.0, 12.0, 0.2), ybound=(-12.0, 12.0, 0.2), zbound=(-0.4, 2.2, 0.2))

cam_order = dict(
    pinhole=["CAM_FRONT_WIDE", "CAM_FRONT_NARROW", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
             "CAM_REAR_LEFT", "CAM_REAR_RIGHT", "CAM_REAR"],
    fisheye=["FISHEYE_FRONT", "FISHEYE_LEFT", "FISHEYE_REAR", "FISHEYE_RIGHT"],
)
img_sizes = dict(pinhole=(256, 448), fisheye=(320, 512))  # 都要能被 32 整除

cam_groups = dict(
    pinhole=dict(camera_model=dict(type="Pinhole"), depth_bins=(0.5, 18.5, 0.25), stride=16),
    fisheye=dict(camera_model=dict(type="KannalaBrandt", max_theta=1.70), depth_bins=(0.2, 16.2, 0.25), stride=16),
)

bev_seg_classes = ["empty", "drivable", "slot_line", "curb", "pillar", "vehicle", "wheel_stopper", "other_static"]
pv_seg_classes = ["other", "drivable", "slot_line", "curb", "pillar", "vehicle", "wheel_stopper", "pedestrian"]
pv_det_classes = ["slot_corner", "ground_lock", "open_door", "wheel_stopper"]
det_classes = ["car", "pedestrian", "cyclist", "ground_lock", "wheel_stopper", "cone", "no_parking_sign"]
occ_sem_classes = ["vehicle", "pillar", "curb", "other"]

num_traj_pts = 40
embed_dims = 256

model = dict(
    type="ParkingE2E",
    grid=grid,
    cam_groups=cam_groups,
    img_backbone=dict(type="TorchvisionResNet", depth=18, weights="IMAGENET1K_V1", frozen_stages=-1),
    img_neck=dict(type="SimpleFPN", out_channels=256, in_strides=(8, 16, 32), out_strides=(8, 16)),
    view_transform=dict(type="MultiGroupLSS", in_channels=256, mid_channels=256, voxel_channels=32, bev_channels=128),
    lidar_encoder=dict(type="PillarEncoder", in_channels=4, feat_channels=64, out_channels=64),
    radar_encoder=dict(type="RadarPillarEncoder", in_channels=7, feat_channels=32, out_channels=32),
    nav_encoder=dict(type="NavigationEncoder", embed_dims=embed_dims, ego_dim=8, max_route_pts=32),
    fuser=dict(type="ConvFuser", in_channels=(128, 64, 32), out_channels=128, modality_dropout=(0.0, 0.1, 0.1)),
    bev_temporal=dict(type="TemporalAlignFuser", channels=128, num_history=3, dim=2),
    voxel_temporal=dict(type="TemporalAlignFuser", channels=32, num_history=3, dim=3),
    bev_encoder=dict(type="ResBEVEncoder", in_channels=128, out_channels=128),
    god_encoder=dict(type="GODVoxelEncoder", voxel_channels=32, bev_channels=128, out_channels=32),
    heads=dict(
        pv_depth=dict(type="PVDepthHead", groups=("pinhole", "fisheye")),
        pv_seg=dict(type="PVSegHead", in_channels=256, num_classes=len(pv_seg_classes),
                    loss=dict(type="CrossEntropyDiceLoss", num_classes=len(pv_seg_classes),
                              class_weight=[0.5, 0.5, 3.0, 2.0, 1.5, 1.0, 3.0, 2.0])),
        pv_det=dict(type="PVDetHead", in_channels=256, num_classes=len(pv_det_classes), stride=8),
        bev_seg=dict(type="BEVSegHead", in_channels=128, num_classes=len(bev_seg_classes), upsample=4, empty_index=0,
                     loss=dict(type="CrossEntropyDiceLoss", num_classes=len(bev_seg_classes),
                               class_weight=[0.5, 0.5, 4.0, 2.0, 2.0, 1.0, 4.0, 1.0])),
        bev_det=dict(type="BEVDetHead", in_channels=128, embed_dims=embed_dims, num_queries=200,
                     num_classes=len(det_classes), num_attrs=2, dynamic_classes=(0, 1, 2)),
        occ=dict(type="OccHead", in_channels=32, num_sem=len(occ_sem_classes), pos_weight=2.0),
        occ_flow=dict(type="OccFlowHead", in_channels=32),
        sdf=dict(type="SDFHead", in_channels=32, tau=0.4, outside_weight=0.1, sign_weight=0.0, eikonal_weight=0.0),
        slot=dict(type="SlotHead", in_channels=128, embed_dims=embed_dims, plan_dims=embed_dims, num_queries=32),
        traj=dict(type="FlowMatchingTrajHead", bev_channels=128, voxel_channels=32, embed_dims=embed_dims,
                  num_heads=8, depth=3, num_pts=num_traj_pts, num_samples=8, sample_steps=8,
                  collision_weight=0.1, clearance=(0.30, 0.30, 0.20),
                  vehicle=dict(length=4.9, width=1.9, rear_overhang=1.0, heights=(0.3, 0.8, 1.4), spacing=0.25)),
    ),
    loss_weights=dict(traj=1.0, slot=1.0, bev_det=0.5, bev_seg=0.5, occ=0.4, occ_flow=0.2, sdf=0.3,
                      pv_det=0.2, pv_seg=0.2, pv_depth=0.2),
    # 部署图：PV 解码、BEV 检测、occ/flow 解码默认关；车位、BEV 分割（车位显示）、轨迹留。
    # SDF 头不在列表里也会被轨迹头的拒绝器在线查询。
    deploy_heads=["slot", "traj", "bev_seg"],
)

data_root = "data/parking/"
train_pipeline = [
    dict(type="LoadMultiViewImages", cam_order=cam_order, data_root=data_root),
    dict(type="LoadPoints", key="lidar", dims=4, data_root=data_root),
    dict(type="LoadPoints", key="radar", dims=7, data_root=data_root),
    dict(type="LoadAnnotations", data_root=data_root),
    dict(type="ResizeCropMultiView", sizes=img_sizes, scale_jitter=(1.0, 1.1), train=True),
    dict(type="DepthFromLidar", camera_models={g: c["camera_model"] for g, c in cam_groups.items()}),
    dict(type="SampleGODQueries", num_occ=16384, num_sdf=16384, num_flow=4096, tau=0.4),
    dict(type="FormatBundle"),
]
history_pipeline = [
    dict(type="LoadMultiViewImages", cam_order=cam_order, data_root=data_root),
    dict(type="LoadPoints", key="lidar", dims=4, data_root=data_root),
    dict(type="LoadPoints", key="radar", dims=7, data_root=data_root),
    dict(type="ResizeCropMultiView", sizes=img_sizes, train=False),
    dict(type="FormatBundle", history=True),
]
train_dataset = dict(type="ParkingE2EDataset", ann_file=data_root + "infos_train.pkl", pipeline=train_pipeline,
                     history_pipeline=history_pipeline, num_history=3, history_stride=2)
data_loader = dict(batch_size=2, num_workers=6, shuffle=True)

optimizer = dict(type="AdamW", lr=2e-4, weight_decay=0.01, paramwise=dict(img_backbone=0.1))
scheduler = dict(type="WarmupCosine", warmup_iters=500, min_ratio=1e-2)
train_cfg = dict(
    epochs=24, log_interval=20, grad_clip=35.0, amp=True, work_dir="work_dirs/parking_e2e_r18",
    # 第一段只开辅助头，把深度、车位线、近场占据学稳；第二段加规划
    stages=[dict(start_epoch=0, loss_weights=dict(traj=0.0, slot=0.0)),
            dict(start_epoch=8, loss_weights=dict(traj=1.0, slot=1.0))],
)
