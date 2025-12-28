"""MMSeg wrapper around the RADSeg encoder"""

import sys
import os

from mmengine.structures import PixelData
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.models.segmentors import BaseSegmentor
from mmseg.registry import MODELS

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
from radseg.radseg import RADSegEncoder  

def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = list(), list()
    for idx in range(num_cls):
        names_i = name_sets[idx].split(', ')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices

@MODELS.register_module()
class RADSegSegmentation(BaseSegmentor):
    def __init__(self, name_path, **kwargs):
        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            rgb_to_bgr=True)
        
        super().__init__(data_preprocessor=data_preprocessor)
        
        cls_names, cls_idxs = get_cls_idx(name_path)
        self.model = RADSegEncoder(classes=cls_names, **kwargs)
        # MMSeg will control normalization
        self.model.model.make_preprocessor_external()

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

        seg_pred,seg_probs = self.model._get_seg_data(inputs, img_size)

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