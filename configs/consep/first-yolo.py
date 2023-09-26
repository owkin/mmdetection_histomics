_base_ = '../yolo/yolov3_d53_8xb8-ms-608-273e_coco.py'

model = dict(
    bbox_head=dict(
        num_classes=7,
    )
)

# Modify dataset related settings
data_root = '/home/owkin/project/datasets/consep/maskrcnn_method_ignore_size_224_step_size_112/'
metainfo = {
    'classes': ('other', 'inflammatory', 'healthy epithelial', 'dysplastic/malignant epithelial', 'fibroblast', 'muscle', 'endothelial')
}

input_size = (320, 320)
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    # `mean` and `to_rgb` should be the same with the `preprocess_cfg`
    dict(type='Expand', mean=[0, 0, 0], to_rgb=True, ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', scale=input_size, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=input_size, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    dataset=dict(
        pipeline=train_pipeline,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='new_train/annotation_coco.json',
        data_prefix=dict(img='tiles/')))

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
                      outfile_prefix='/home/owkin/project/work_dirs/first-yolo/test_17')

load_from = "https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_320_273e_coco/yolov3_d53_320_273e_coco-421362b6.pth"
work_dir = '/home/owkin/project/work_dirs/first-yolo'
