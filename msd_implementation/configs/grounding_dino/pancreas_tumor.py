_base_ = 'grounding_dino_swin-t_pretrain_obj365_goldg.py'

data_root = 'data/MSD_pancreas/'

# Deux classes pour matcher le checkpoint finetune
metainfo = dict(
    classes=('pancreas', 'tumor'),
    palette=[(0, 255, 0), (255, 0, 0)],
)

label_map_path = 'data/MSD_pancreas/tumor_label_map.json'

model = dict(
    backbone=dict(frozen_stages=0),
    bbox_head=dict(
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=3.0,
        )
    ),
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ),
    dict(
        type='RandomSamplingNegPos',
        tokenizer_name='bert-base-uncased',
        num_sample_negative=3,
        label_map_file=label_map_path,
    ),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'flip',
            'flip_direction',
            'text',
            'custom_entities',
        ),
    ),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities'))
]

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
        return_classes=True
    )
)

# Override test_dataloader (sinon _base_ pointe vers COCO 2017 avec scale 800x1333)
# C'est celui-ci que mmdet.inference_detector lit pour construire son pipeline.
test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        return_classes=True,
        ann_file='test.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline
    )
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val.json',
    metric='bbox',
    classwise=True,
    proposal_nums=(100, 300, 1000)
)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test.json',
    metric='bbox'
)

fp16 = dict(loss_scale='dynamic')

optim_wrapper = dict(
    type='OptimWrapper',
    # LR bas pour eviter de degrader le backbone degelé
    optimizer=dict(type='AdamW', lr=0.00001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    accumulative_counts=2
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=25, val_interval=1)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=1,
        save_best='coco/bbox_mAP',
        rule='greater',
        save_optimizer=False,
    ),
    logger=dict(type='LoggerHook', interval=50),
    visualization=dict(type='DetVisualizationHook')
)

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        patience=6,
        min_delta=0.005
    )
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ],
    name='visualizer'
)

load_from = 'models_weights/grounding_dino_swin-t_pretrain_obj365_goldg_20231122_132602-4ea751ce.pth'
