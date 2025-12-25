"""Script to run all RADSeg evals on all datasets on all resolutions"""

import os

datasets = [
    "ade20k",
    "cocostuff",
    "voc20",
    "city_scapes",
    "context59",
]

resolutions = ["low_res", "mid_res", "high_res"]

configs = {
    res: [
        f"./configs_mmseg/{res}_configs/cfg_{ds}.py"
        for ds in datasets
    ]
    for res in resolutions
}

model = "c-radio_v3-b"
lang_model = "siglip2"
scra_scaling = 10.0
scga_scaling = 10.0
base_output_path = "results/radseg_segmentation"
sam_refine = False

for res, configs_list in configs.items():
    output_path = os.path.join(base_output_path, res)
    for config in configs_list:
        print(f"Running {config}")
        os.system(
            f"python eval.py --config {config} "
            f"--model_version {model} --lang_model {lang_model} "
            f"--work-dir {output_path} "
            f"--scra_scaling {scra_scaling} --scga_scaling {scga_scaling} "
            f"--sam_refine {sam_refine}"
        )
