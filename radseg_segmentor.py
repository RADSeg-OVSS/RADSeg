import math
import os
import time

import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.structures import PixelData
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.models.segmentors import BaseSegmentor
from mmseg.registry import MODELS
from encoders import RADSegEncoder  

@MODELS.register_module()
class RADSegSegmentation(BaseSegmentor):
    def __init__(self, name_path, model_version = "radio_v3-b", lang_model = "siglip2", device=torch.device('cuda'),
                prob_thd=0.0, prompt_denoising_thresh=0.5, slide_stride=112, slide_crop=224, compile=False, scga_scaling=10.0, 
                scra_scaling=10.0, amp = False,sam_ckpt= '/path/to/sam_h_ckpt',coarse_thresh=0.10,minimal_area=225,
                sam_mask_coff=0.005, sam_target_size = 1024,sam_model_type='vit_h', sam_refinement=False, **kwargs):

        data_preprocessor = SegDataPreProcessor(mean=[122.771, 116.746, 104.094], std=[68.501, 66.632, 70.323], rgb_to_bgr=True)
        super().__init__(data_preprocessor=data_preprocessor)
        
        use_sliding_window = slide_crop > 0    

        self.model = RADSegEncoder(
            device=device,
            name_path=name_path,
            model_version=model_version,
            lang_model=lang_model,
            compile=compile,
            amp=amp,
            slide_crop=slide_crop,
            slide_stride=slide_stride,
            scga_scaling=scga_scaling,
            scra_scaling=scra_scaling,
            sam_ckpt=sam_ckpt,
            coarse_thresh=coarse_thresh,
            minimal_area=minimal_area,
            sam_mask_coff=sam_mask_coff,
            sam_target_size=sam_target_size,
            sam_model_type=sam_model_type,
            sam_refinement=sam_refinement,
            prompt_denoising_thresh=prompt_denoising_thresh,
            **kwargs
        )


    def predict(self, inputs, data_samples):

        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            batch_img_metas = [dict(
                ori_shape=inputs.shape[2:],
                img_shape=inputs.shape[2:],
                pad_shape=inputs.shape[2:],
                padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        img_size = batch_img_metas[0]['ori_shape']

        seg_pred,seg_probs = self.model.get_seg_data(inputs, img_size)


        data_samples[0].set_data({
                'seg_logits': PixelData(**{'data': seg_probs}),
                'pred_sem_seg': PixelData(**{'data': seg_pred})
            })

        return data_samples

    def _forward(data_samples):
        pass

    def inference(self, img, batch_img_metas):
        pass

    def encode_decode(self, inputs, batch_img_metas):
        pass

    def extract_feat(self, inputs):
        pass

    def loss(self, inputs, data_samples):
        pass