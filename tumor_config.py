_base_ = 'grounding_dino_swin-t_pretrain_obj365_goldg.py'

# 1. Nouveaux chemins et métadonnées
data_root = 'data/MSD_pancreas/'

# On ne déclare que la classe cible. La couleur rouge (255, 0, 0) est utilisée pour l'affichage.
metainfo = dict(classes=('tumor',), palette=[(255, 0, 0)])
label_map_path = 'data/MSD_pancreas/tumor_label_map.json'

# 2. Architecture (Maintien des optimisations)
model = dict(
    backbone=dict(frozen_stages=2)
)

# 3. Pipelines avec distorsion photométrique
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(800, 1333), keep_ratio=True),
    dict(type='PhotoMetricDistortion', 
         brightness_delta=32, 
         contrast_range=(0.5, 1.5), 
         saturation_range=(0.5, 1.5), 
         hue_delta=18),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomSamplingNegPos',
        tokenizer_name='bert-base-uncased',
        num_sample_negative=0,
        label_map_file=label_map_path),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(800, 1333), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities'))
]

# 4. Dataloaders pointant vers les JSON divisés
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train.json',
        data_prefix=dict(img=''), # Chemin ajusté selon ton JSON
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        return_classes=True,
        ann_file='val.json',
        data_prefix=dict(img=''), # Chemin ajusté selon ton JSON
        test_mode=True,
        pipeline=test_pipeline
    )
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val.json',
    metric='bbox'
)

# 5. Paramètres d'entraînement et Early Stopping
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.00002, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2)
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=10, val_interval=1)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', 
        interval=1, 
        max_keep_ckpts=1, 
        save_optimizer=False,
        save_best='coco/bbox_mAP'
    ),
    logger=dict(type='LoggerHook', interval=50),
    visualization=dict(type='DetVisualizationHook')
)

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        patience=3,
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

load_from = 'grounding_dino_swin-t_pretrain_obj365_goldg_20231122_132602-4ea751ce.pth'