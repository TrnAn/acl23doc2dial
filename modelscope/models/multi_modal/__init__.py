# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:

    from .clip import CLIPForMultiModalEmbedding
    from .gemm import GEMMForMultiModalEmbedding
    from .rleg import RLEGForMultiModalEmbedding
    from .team import TEAMForMultiModalSimilarity
    from .diffusion import DiffusionForTextToImageSynthesis
    from .mmr import VideoCLIPForMultiModalEmbedding
    from .mplug_for_all_tasks import MPlugForAllTasks, HiTeAForAllTasks
    from .ofa_for_all_tasks import OfaForAllTasks
    from .ofa_for_text_to_image_synthesis_model import \
        OfaForTextToImageSynthesis
    from .multi_stage_diffusion import \
        MultiStageDiffusionForTextToImageSynthesis
    from .vldoc import VLDocForDocVLEmbedding
    from .video_synthesis import TextToVideoSynthesis
    from .efficient_diffusion_tuning import EfficientStableDiffusion

else:
    _import_structure = {
        'clip': ['CLIPForMultiModalEmbedding'],
        'diffusion': ['DiffusionForTextToImageSynthesis'],
        'gemm': ['GEMMForMultiModalEmbedding'],
        'rleg': ['RLEGForMultiModalEmbedding'],
        'team': ['TEAMForMultiModalSimilarity'],
        'mmr': ['VideoCLIPForMultiModalEmbedding'],
        'mplug_for_all_tasks': ['MPlugForAllTasks', 'HiTeAForAllTasks'],
        'ofa_for_all_tasks': ['OfaForAllTasks'],
        'ofa_for_text_to_image_synthesis_model':
        ['OfaForTextToImageSynthesis'],
        'multi_stage_diffusion':
        ['MultiStageDiffusionForTextToImageSynthesis'],
        'vldoc': ['VLDocForDocVLEmbedding'],
        'video_synthesis': ['TextToVideoSynthesis'],
        'efficient_diffusion_tuning': ['EfficientStableDiffusion']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
