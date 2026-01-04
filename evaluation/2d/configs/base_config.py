# base configurations
model = dict(
    type='RADSegSegmentation',
    model_version = "c-radio_v3-b",
    lang_model = "siglip2",
    prompt_denoising_thresh=0.5,
    slide_crop=336,
    slide_stride=224,
    amp=False,
    compile=False,
    sam_ckpt = 'sam_vit_h_4b8939.pth'
)

test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=None,
        data_root=None,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        )
)
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    visualization=dict(type='SegVisualizationHook',
                       draw=False, # Set to True to visualize
                       interval=1))
