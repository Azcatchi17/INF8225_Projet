_base_ = 'grounding_dino_swin-t_pretrain_obj365_goldg.py'

# 1. Nouveaux chemins et métadonnées (V4)
data_root = 'data/MSD_pancreas/'

# AJOUT : Les 3 classes pour forcer le signal négatif et l'apprentissage contrastif
metainfo = dict(
    classes=('pancreatic tumor', 'normal pancreas', 'bowel gas'),
    palette=[(255, 0, 0), (0, 255, 0), (0, 0, 255)]
)
label_map_path = 'data/MSD_pancreas/tumor_label_map.json'

# 2. Architecture
model = dict(
    backbone=dict(frozen_stages=2),
    # AJOUT : Augmentation du poids de la loss de classification (1.5)
    bbox_head=dict(
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.5 
        )
    )
)

# 3. Pipelines
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # MODIFICATION : Résolution CT native pour éviter l'interpolation destructrice
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    # SUPPRESSION : PhotoMetricDistortion retiré car inadapté aux scans CT
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomSamplingNegPos',
        tokenizer_name='bert-base-uncased',
        # MODIFICATION : Hard negative mining activé
        num_sample_negative=5,
        label_map_file=label_map_path),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # MODIFICATION : Alignement de la résolution de test
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities'))
]

# 4. Dataloaders (Noms de fichiers inchangés)
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
        pipeline=train_pipeline
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
        return_classes=True,
        ann_file='val.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline
    )
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val.json',
    metric='bbox'
)

# 5. Paramètres d'entraînement
fp16 = dict(loss_scale='dynamic')

optim_wrapper = dict(
    type='OptimWrapper',
    # MODIFICATION : LR augmenté pour accélérer la convergence sur peu d'époques
    optimizer=dict(type='AdamW', lr=0.00003, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    # MODIFICATION : Réduction de l'accumulation pour stabiliser le batch effectif
    accumulative_counts=2
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=25, val_interval=1)

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
        # MODIFICATION : Patience légèrement augmentée pour tolérer les fluctuations
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

load_from = 'grounding_dino_swin-t_pretrain_obj365_goldg_20231122_132602-4ea751ce.pth'