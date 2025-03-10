# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict, Union

import torch.cuda

from modelscope.metainfo import Models
from modelscope.models.base import Tensor
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .nafnet.NAFNet_arch import NAFNet, PSNRLoss

logger = get_logger()
__all__ = ['NAFNetForImageDenoise']


@MODELS.register_module(Tasks.image_denoising, module_name=Models.nafnet)
class NAFNetForImageDenoise(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the image denoise model from the `model_dir` path.

        Args:
            model_dir (str): the model path.

        """
        super().__init__(model_dir, *args, **kwargs)
        self.model_dir = model_dir
        self.config = Config.from_file(
            os.path.join(self.model_dir, ModelFile.CONFIGURATION))
        model_path = os.path.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        self.model = NAFNet(**self.config.model.network_g)
        self.loss = PSNRLoss()
        self.model = self._load_pretrained(self.model, model_path)

    def _train_forward(self, input: Tensor,
                       target: Tensor) -> Dict[str, Tensor]:
        preds = self.model(input)
        return {'loss': self.loss(preds, target)}

    def _inference_forward(self, input: Tensor) -> Dict[str, Tensor]:
        return {'outputs': self.model(input).clamp(0, 1)}

    def _evaluate_postprocess(self, input: Tensor,
                              target: Tensor) -> Dict[str, list]:
        preds = self.model(input)
        preds = list(torch.split(preds.clamp(0, 1), 1, 0))
        targets = list(torch.split(target.clamp(0, 1), 1, 0))

        return {'pred': preds, 'target': targets}

    def forward(self, inputs: Dict[str,
                                   Tensor]) -> Dict[str, Union[list, Tensor]]:
        """return the result by the model

        Args:
            inputs (Tensor): the preprocessed data

        Returns:
            Dict[str, Tensor]: results
        """
        if self.training:
            return self._train_forward(**inputs)
        elif 'target' in inputs:
            return self._evaluate_postprocess(**inputs)
        else:
            return self._inference_forward(**inputs)
