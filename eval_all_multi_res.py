import os

configs_list_low_res = [
     './configs_mmseg/low_res_configs/cfg_ade20k.py',
     './configs_mmseg/low_res_configs/cfg_cocostuff.py',
    './configs_mmseg/low_res_configs/cfg_voc20.py',
     './configs_mmseg/low_res_configs/cfg_city_scapes.py',
     './configs_mmseg/low_res_configs/cfg_context59.py',
]

configs_list_mid_res = [
     './configs_mmseg/mid_res_configs/cfg_ade20k.py',
     './configs_mmseg/mid_res_configs/cfg_cocostuff.py',
    './configs_mmseg/mid_res_configs/cfg_voc20.py',
     './configs_mmseg/mid_res_configs/cfg_city_scapes.py',
     './configs_mmseg/mid_res_configs/cfg_context59.py',
]

configs_list_high_res = [   
     './configs_mmseg/high_res_configs/cfg_ade20k.py',
     './configs_mmseg/high_res_configs/cfg_cocostuff.py',
    './configs_mmseg/high_res_configs/cfg_voc20.py',
     './configs_mmseg/high_res_configs/cfg_city_scapes.py',
     './configs_mmseg/high_res_configs/cfg_context59.py',
]

configs = {
     "low_res": configs_list_low_res,
     "mid_res": configs_list_mid_res,
     "high_res": configs_list_high_res,
}

model,lang_model =  ("c-radio_v3-b","siglip2")
scra_scaling = '10.0'
scga_scaling = "10.0"
base_output_path = "results/radseg_plus_segmentation"
sam_refine = "True"

for res, configs_list in configs.items():
     output_path = os.path.join(base_output_path, res)
     for config in configs_list:
          print(f"Running {config}")
          os.system(f"bash ./dist_test.sh {config} {model} {lang_model} {output_path} {scra_scaling} {scga_scaling} {sam_refine}")