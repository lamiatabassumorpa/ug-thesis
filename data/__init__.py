"""
Data pipeline for Multi-Stream Deepfake Detection.

Datasets:
  DeepfakeDataset  — Custom image dataset (FFHQ/DiffusionDB/GenImage/ForenSynths)
  FFPPDataset      — FaceForensics++ (ICCV 2019), standard CVPR benchmark
  CelebDFDataset   — Celeb-DF v2 (CVPR 2020), cross-dataset evaluation
"""
from .dataset import DeepfakeDataset, create_dataloaders, get_transforms
from .ffpp_dataset import (
    FFPPDataset,
    create_ffpp_dataloaders,
    get_ffpp_transforms,
    aggregate_video_predictions,
    MANIPULATION_TYPES,
)
from .celebdf_dataset import CelebDFDataset, create_celebdf_testloader

__all__ = [
    # Original dataset
    "DeepfakeDataset",
    "create_dataloaders",
    "get_transforms",
    # FaceForensics++
    "FFPPDataset",
    "create_ffpp_dataloaders",
    "get_ffpp_transforms",
    "aggregate_video_predictions",
    "MANIPULATION_TYPES",
    # Celeb-DF v2
    "CelebDFDataset",
    "create_celebdf_testloader",
]
