
import numpy as np
import torch
from torch import Tensor as T


def xyxy_to_cxcywh(bbox: np.ndarray | T):
    ndim = len(bbox.shape)
    if ndim == 1:
        bbox = bbox[None, :]
    cxcy = (bbox[..., :2] + bbox[..., 2:]) / 2
    wh = bbox[..., 2:] - bbox[..., :2]
    if isinstance(cxcy, T) and isinstance(wh, T):
        new_bbox = torch.cat([cxcy, wh], dim=-1)
    else:
        new_bbox = np.concatenate([cxcy, wh], axis=-1)
    if ndim == 1:
        new_bbox = new_bbox[0]
    return new_bbox


def cxcywh_to_xyxy(bbox: np.ndarray | T):
    ndim = len(bbox.shape)
    if ndim == 1:
        bbox = bbox[None, :]
    xy = bbox[..., :2] - bbox[..., 2:] / 2
    if isinstance(xy, np.ndarray):
        xy = np.maximum(xy, 0)
    elif isinstance(xy, T):
        xy = torch.max(xy, torch.zeros_like(xy))
    wh = bbox[..., 2:]
    if isinstance(xy, T) and isinstance(wh, T):
        new_bbox = torch.cat([xy, xy + wh], dim=-1)
    else:
        new_bbox = np.concatenate([xy, xy + wh], axis=-1)
    if ndim == 1:
        new_bbox = new_bbox[0]
    return new_bbox


def xyxy_to_xywh(bbox: np.ndarray | T):
    ndim = len(bbox.shape)
    if ndim == 1:
        bbox = bbox[None, :]
    xy = bbox[..., :2]
    wh = bbox[..., 2:] - bbox[..., :2]
    if isinstance(xy, T) and isinstance(wh, T):
        new_bbox = torch.cat([xy, wh], dim=-1)
    else:
        new_bbox = np.concatenate([xy, wh], axis=-1)
    if ndim == 1:
        new_bbox = new_bbox[0]
    return new_bbox


def xywh_to_xyxy(bbox: np.ndarray | T):
    ndim = len(bbox.shape)
    if ndim == 1:
        bbox = bbox[None, :]
    xy = bbox[..., :2]
    wh = bbox[..., 2:]
    if isinstance(xy, T) and isinstance(wh, T):
        new_bbox = torch.cat([xy, xy + wh], dim=-1)
    else:
        new_bbox = np.concatenate([xy, xy + wh], axis=-1)
    if ndim == 1:
        new_bbox = new_bbox[0]
    return new_bbox
