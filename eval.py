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
    parser.add_argument('--type', default='', help='segmentor type')
    parser.add_argument('--sim_scale', default='', help='similarity scale')
    parser.add_argument('--agg_beta', default='', help='aggregation beta scale')
    parser.add_argument('--agg_gamma', default='', help='aggregation gamma scale')
    parser.add_argument('--sam_refine', action='store_true')

    args = parser.parse_args()
    return args


def trigger_visualization_hook(cfg, show_dir):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        visualization_hook['draw'] = True
        visualizer = cfg.visualizer
        visualizer['save_dir'] = show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks. refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')
    cfg.model['pamr_steps'] = 50
    cfg.model['pamr_stride'] = [1, 2, 4, 8, 12, 24]
    return cfg


def safe_set_arg(cfg, arg, name, func=lambda x: x):
    if arg != '':
        cfg.model[name] = func(arg)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir

    if len(args.model_version) > 0:
        cfg['model']['model_version'] = args.model_version

    if len(args.lang_model) > 0:
        cfg['model']['lang_model'] = args.lang_model

    if len(args.type) > 0:
        cfg['model']['type'] = args.type

    if len(args.sim_scale) > 0:
        cfg['model']['sim_scale'] = int(args.sim_scale)

    if len(args.agg_beta) > 0:
        cfg['model']['agg_beta'] = float(args.agg_beta)

    if len(args.agg_gamma) > 0:
        cfg['model']['agg_gamma'] = float(args.agg_gamma)

    cfg.model.sam_refinement = args.sam_refine

    runner = Runner.from_cfg(cfg)

    results = runner.test()

    results.update({'Segmentation Type': cfg.model.type,
                    'Model Version': cfg.model.model_version,
                    'Dataset': cfg.dataset_type,
                })

    with open(os.path.join(cfg.work_dir, 'results.txt'), 'a') as f:
        f.write(os.path.basename(args.config).split('.')[0] + '\n')
        for k, v in results.items():
            f.write(k + ': ' + str(v) + '\n')

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    main()