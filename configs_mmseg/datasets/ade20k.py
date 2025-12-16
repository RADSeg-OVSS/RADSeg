_base_ = '../base_config.py'

# dataset settings
dataset_type = 'ADE20KDataset'
data_root = '/ocean/projects/cis220039p/mdt2/djariwala/semseg_datasets/ADEChallengeData2016'

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation')
    )
)

# model settings
model = dict(
    name_path='./configs_mmseg/cls_ade20k.txt',
    # SAM params
    refine_neg_cos = False,
    coarse_thresh=0.05,
)
