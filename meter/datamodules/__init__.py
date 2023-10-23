from .vg_caption_datamodule import VisualGenomeCaptionDataModule
from .f30k_caption_karpathy_datamodule import F30KCaptionKarpathyDataModule
from .coco_caption_karpathy_datamodule import CocoCaptionKarpathyDataModule
from .conceptual_caption_datamodule import ConceptualCaptionDataModule
from .sbu_datamodule import SBUCaptionDataModule
from .vqav2_datamodule import VQAv2DataModule
from .nlvr2_datamodule import NLVR2DataModule
from .snli_datamodule import SNLIDataModule
from .sinc_datamodule import SINCDataModule # NEW
from .open_mi_datamodule import OPENMIDataModule # NEW
from .fast_vqa_datamodule import FASTVQADataModule # NEW

_datamodules = {
    "vg": VisualGenomeCaptionDataModule,
    "f30k": F30KCaptionKarpathyDataModule,
    "coco": CocoCaptionKarpathyDataModule,
    "gcc": ConceptualCaptionDataModule,
    "sbu": SBUCaptionDataModule,
    "vqa": VQAv2DataModule,
    "nlvr2": NLVR2DataModule,
    "snli": SNLIDataModule,
    "sinc": SINCDataModule, # NEW
    "open_mi": OPENMIDataModule, # NEW
    "fast_vqa": FASTVQADataModule, # NEW
}
