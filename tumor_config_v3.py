_base_ = 'grounding_dino_swin-t_pretrain_obj365_goldg.py'

data_root = 'data/MSD_pancreas/'

metainfo = dict(
    classes=('pancreas', 'tumor'),
    palette=[(0, 255, 0), (255, 0, 0)]
)

label_map_path = 'data/MSD_pancreas/tumor_label_map.json'

model = dict(
    # MODIFICATION #1 : On dégèle tout. Le modèle va réapprendre ses filtres de 
    # bas niveau pour s'adapter au contraste spécifique du scanner CT.
    backbone=dict(frozen_stages=0),
    bbox_head=dict(
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25, # Retour à l'équilibre sain
            loss_weight=3.0 
        )
    )
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5)
    ),
    
    dict(
        type='RandomSamplingNegPos',
        tokenizer_name='bert-base-uncased',
        # MODIFICATION #2 : Baisse des distracteurs de 10 à 3
        num_sample_negative=3,
        label_map_file=label_map_path),
        
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
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
    batch_size=2,
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

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val.json',
    metric='bbox',
    classwise=True,
    proposal_nums=(100, 300, 1000)
)

fp16 = dict(loss_scale='dynamic')
optim_wrapper = dict(
    type='OptimWrapper',
    # MODIFICATION #3 : Baisse du Learning Rate (de 3e-5 à 1e-5). 
    # Vital, car on a dégelé le backbone. Si le LR est trop haut, il va détruire ses poids.
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
        save_optimizer=False 
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

# Poids de base pour un nouveau départ sain
load_from = 'grounding_dino_swin-t_pretrain_obj365_goldg_20231122_132602-4ea751ce.pth'