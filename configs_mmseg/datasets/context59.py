_base_ = '../base_config.py'

# dataset settings
dataset_type = 'PascalContext59Dataset'
data_root = '/ocean/projects/cis220039p/mdt2/djariwala/semseg_datasets/VOCdevkit/VOC2010'

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClassContext'),
        ann_file='ImageSets/SegmentationContext/val.txt'
    )
)

# model settings
model = dict(
    name_path='./configs_mmseg/cls_context59.txt',
    # SAM params
    coarse_thresh = 0.15,
)
