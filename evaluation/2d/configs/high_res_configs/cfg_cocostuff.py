_base_ = '../datasets/coco_stuff.py'



test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1334, 896), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(dataset=dict(pipeline=test_pipeline))