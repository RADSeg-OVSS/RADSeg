"""MMSeg wrapper around the RADSeg encoder"""

from mmengine.structures import PixelData
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.models.segmentors import BaseSegmentor
from mmseg.registry import MODELS
from encoders import RADSegEncoder  

@MODELS.register_module()
class RADSegSegmentation(BaseSegmentor):
    def __init__(self, **kwargs):
        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            rgb_to_bgr=True)
        
        super().__init__(data_preprocessor=data_preprocessor)
        self.model = RADSegEncoder(**kwargs)

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