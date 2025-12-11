_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs_mmseg/cls_voc20.txt',
    # SAM params
    coarse_thresh=0.2,
    cos_fac=0.0,
)

# dataset settings
dataset_type = 'PascalVOC20Dataset'
data_root = '/ocean/projects/cis220039p/mdt2/djariwala/semseg_datasets/VOCdevkit/VOC2012'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2016, 672), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/val.txt',
        pipeline=test_pipeline))