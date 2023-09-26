# The new config inherits a base config to highlight the necessary modification
_base_ = '../mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
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

train_dataloader = dict(
    batch_size=1,
    dataset=dict(
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
                      outfile_prefix='/home/owkin/project/work_dirs/baseline-mask-rcnn/test_1')

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
# load_from = '/home/owkin/mmdetection_histomics/work_dirs/first-mask-rcnn-consep/epoch_76.pth'

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300, val_interval=1)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=200,
        by_epoch=True,
        milestones=[100, 200],
        gamma=0.1)
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001))

work_dir = '/home/owkin/project/work_dirs/baseline-mask-rcnn'
