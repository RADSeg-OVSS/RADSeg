_base_ = '../datasets/voc20.py'



test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2016, 672), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(dataset=dict(pipeline=test_pipeline))