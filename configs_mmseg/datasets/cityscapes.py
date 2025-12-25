_base_ = '../base_config.py'

# dataset settings
dataset_type = 'CityscapesDataset'
data_root = 'datasets/cityscapes'

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val')
    )
)

# model settings
model = dict(
    name_path='./configs_mmseg/cls_city_scapes.txt',
    # SAM params
    cos_fac = 3.0,
    refine_neg_cos = False,
    coarse_thresh=0.10,
)
