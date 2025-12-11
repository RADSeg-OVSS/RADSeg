_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs_mmseg/cls_city_scapes.txt',
    # SAM params
    cos_fac = 3.0,
    refine_neg_cos = False,
    coarse_thresh=0.10,
)

# dataset settings
dataset_type = 'CityscapesDataset'
data_root = '/ocean/projects/cis220039p/mdt2/djariwala/semseg_datasets/cityscapes'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='Resize', scale=(512, 512)),
    dict(type='Resize', scale=(2688, 1344), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth does not need to do resize data transform
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
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        pipeline=test_pipeline))