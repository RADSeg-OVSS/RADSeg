import argparse
import os
from mmengine.config import Config
from mmengine.runner import Runner
import custom_datasets
import radseg_segmentor
import torch
import os 

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation with MMSeg')
    parser.add_argument('--config', default='')
    parser.add_argument('--work-dir', default='./work_logs/')
    parser.add_argument('--show-dir', default='', help='directory to save visualization images')
    parser.add_argument('--model_version', default='', help='radio model version')
    parser.add_argument('--lang_model', default='', help='language model')
    parser.add_argument('--scga_scaling', default='', help='scga scale')
    parser.add_argument('--scra_scaling', default='', help='scra scale')
    parser.add_argument('--sam_refine', action='store_true')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir

    if len(args.model_version) > 0:
        cfg['model']['model_version'] = args.model_version

    if len(args.lang_model) > 0:
        cfg['model']['lang_model'] = args.lang_model

    if len(args.scga_scaling) > 0:
        cfg['model']['scga_scaling'] = float(args.scga_scaling)

    if len(args.scra_scaling) > 0:
        cfg['model']['scra_scaling'] = float(args.scra_scaling)

    cfg.model.sam_refinement = args.sam_refine

    runner = Runner.from_cfg(cfg)

    results = runner.test()

    results.update({'Model Version': cfg.model.model_version,
                    'Dataset': cfg.dataset_type,
                    'Sam Refinement': cfg.model.sam_refinement,
                })

    with open(os.path.join(cfg.work_dir, 'results.txt'), 'a') as f:
        f.write(os.path.basename(args.config).split('.')[0] + '\n')
        for k, v in results.items():
            f.write(k + ': ' + str(v) + '\n')

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    main()