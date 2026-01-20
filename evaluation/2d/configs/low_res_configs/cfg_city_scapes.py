_base_ = '../datasets/cityscapes.py'

model = dict(
    slide_stride=112,
    slide_crop=224
)


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 560), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(dataset=dict(pipeline=test_pipeline))