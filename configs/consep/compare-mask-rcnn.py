# The new config inherits a base config to highlight the necessary modification
_base_ = '../mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_roi_extractor=dict(
            roi_layer=dict(sampling_ratio=2)
        ),
        mask_roi_extractor=dict(
            roi_layer=dict(sampling_ratio=2)
        ),
        bbox_head=dict(num_classes=7),
        mask_head=dict(num_classes=7)
    ),
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
    ]

# Modify dataset related settings
data_root = '/home/owkin/project/datasets/consep/maskrcnn_method_ignore_size_224_step_size_112/'
metainfo = {
    'classes': ('other', 'inflammatory', 'healthy epithelial', 'dysplastic/malignant epithelial', 'fibroblast', 'muscle', 'endothelial')
}

train_dataloader = dict(
    batch_size=5,
    dataset=dict(
        pipeline=train_pipeline,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='new_train/annotation_coco.json',
        data_prefix=dict(img='tiles/')))
# val_dataloader = None
# val_cfg = None
# test_cfg = None
# test_dataloader = None
val_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='new_val/annotation_coco.json',
        data_prefix=dict(img='tiles/')))
test_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='new_test/annotation_coco.json',
        data_prefix=dict(img='tiles/')))

# Modify metric related settings
# val_evaluator = None
# test_evaluator = None
val_evaluator = dict(ann_file=data_root + 'new_val/annotation_coco.json')
test_evaluator = dict(ann_file=data_root + 'new_test/annotation_coco.json')

# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
# load_from = '/home/owkin/mmdetection_histomics/work_dirs/first-mask-rcnn-consep/epoch_76.pth'
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)
# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=200,
#         by_epoch=True,
#         milestones=[10, 100],
#         gamma=0.1)
# ]
param_scheduler = None

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=0.0001,  weight_decay=0.00001))

work_dir = '/home/owkin/project/work_dirs/compare-mask-rcnn'