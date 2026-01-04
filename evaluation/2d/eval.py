import argparse
import os

import torch

from mmengine.config import Config
from mmengine.runner import Runner
# Next 2 imports needed for mmseg registration
import custom_datasets
import radseg_segmentor

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation with MMSeg')
    parser.add_argument('--config',type=str, default='configs/mid_res_configs/cfg_voc20.py')
    parser.add_argument('--work-dir',type=str, default='./work_logs/')
    parser.add_argument('--model_version',type=str, default='c-radio_v3-b', help='radio model version')
    parser.add_argument('--lang_model',type=str, default='siglip2', help='language model')
    parser.add_argument('--scra_scaling',type=float, default=10.0, help='SCRA scale')
    parser.add_argument('--scga_scaling',type=float, default=10.0, help='SCGA scale')
    parser.add_argument('--sam_refine', action='store_true', help='RADIO-SAM refinement')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir

    cfg.model.model_version = args.model_version
    cfg.model.lang_model = args.lang_model
    cfg.model.scga_scaling = args.scga_scaling
    cfg.model.scra_scaling = args.scra_scaling

    cfg.model.sam_refinement = args.sam_refine

    runner = Runner.from_cfg(cfg)

    # Inject class names into model if no prompt file path is provided
    if not hasattr(cfg.model, "name_path"):
        runner.model.model._init_semseg_prompts(runner.test_dataloader.dataset.metainfo["classes"])

    results = runner.test()

    results.update({'Model Version': cfg.model.model_version,
                    'Dataset': cfg.dataset_type,
                    'Sam Refinement': cfg.model.sam_refinement})

    with open(os.path.join(cfg.work_dir, 'results.txt'), 'a') as f:
        f.write(os.path.basename(args.config).split('.')[0] + '\n')
        for k, v in results.items():
            f.write(k + ': ' + str(v) + '\n')

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    main()