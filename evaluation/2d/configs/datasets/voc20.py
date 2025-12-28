_base_ = '../base_config.py'

# dataset settings
dataset_type = 'PascalVOC20Dataset'
data_root = 'datasets/VOC2012'

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
        ann_file='ImageSets/Segmentation/val.txt'
    )
)

# model settings
model = dict(
    name_path='./configs/cls_voc20.txt',
    coarse_thresh=0.2,
)
