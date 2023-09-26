# The new config inherits a base config to highlight the necessary modification
_base_ = '../mask_rcnn/mask-rcnn_x101-64x4d_fpn_ms-poly_3x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    # backbone=dict(frozen_stages=4),
    roi_head=dict(
        bbox_head=dict(num_classes=7),
        mask_head=dict(num_classes=7)
    ),
)

# Modify dataset related settings
data_root = '/home/owkin/project/datasets/consep/maskrcnn_method_ignore_size_224_step_size_112/'
metainfo = {
    'classes': ('other', 'inflammatory', 'healthy epithelial', 'dysplastic/malignant epithelial', 'fibroblast', 'muscle', 'endothelial')
}

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='RandomCrop',
        crop_size=(.75, .75),
        crop_type='relative',
    ),
    dict(
        type='PhotoMetricDistortion'
    ),
    dict(
        type='RandomChoiceResize',
        scales=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                (1333, 768), (1333, 800)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        dataset=dict(
            pipeline=train_pipeline,
            data_root=data_root,
            metainfo=metainfo,
            ann_file='new_train/annotation_coco.json',
            data_prefix=dict(img='tiles/'))
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='new_val/annotation_coco.json',
        data_prefix=dict(img='tiles/')))

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='new_test/annotation_coco.json',
        data_prefix=dict(img='tiles/')))


val_evaluator = dict(ann_file=data_root + 'new_val/annotation_coco.json')
test_evaluator = dict(ann_file=data_root + 'new_test/annotation_coco.json',
                      format_only=True,
                      outfile_prefix='/home/owkin/project/work_dirs/big-mask-rcnn/test')

load_from = "https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco_20210526_120447-c376f129.pth"

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[9, 11],
        gamma=0.1)
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))

work_dir = '/home/owkin/project/work_dirs/big-mask-rcnn'