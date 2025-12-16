_base_ = '../base_config.py'

# dataset settings
dataset_type = 'COCOStuffDataset'
data_root = '/ocean/projects/cis220039p/mdt2/djariwala/semseg_datasets/coco_stuff164k'

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/val2017', seg_map_path='annotations/val2017')
    )
)

# model settings
model = dict(
    name_path='./configs_mmseg/cls_coco_stuff.txt',
    # SAM params
    coarse_thresh=0.05,
)
