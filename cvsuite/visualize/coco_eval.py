
import os
from typing import Callable

from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from cvsuite.utils.logging import create_logger
from cvsuite.visualize.utils import read_image, save_image, plot_xywh_bbox


logger = create_logger(__name__)
PairType = tuple[int, list[int], list[int]]
FilterPolicyType = Callable[[COCOeval], list[PairType]]


def default_filter_policy(coco_eval: COCOeval):
    """
    Default filter policy for COCO visualization.
    """
    img_ids = coco_eval.params.imgIds
    pairs: list[tuple[int, list[int], list[int]]] = []
    for img_id in img_ids:
        gt_ids = coco_eval.cocoGt.getAnnIds(img_id)
        dt_ids = coco_eval.cocoDt.getAnnIds(img_id)
        pairs.append((img_id, gt_ids, dt_ids))
    return pairs


def show_image_given_coco_eval(
        coco_eval: COCOeval,
        image_root: str,
        output_root: str,
        filter_policy: FilterPolicyType | None = None):
    """
    Visualize the COCO evaluation results.

    Argument(s):
        coco_eval: The COCO evaluation object.
        image_root: The root of the images. This function will concatenate
            the image root and the filename to find the image.
        output_root: The root to save the visualized images.
        filter_policy: The filter policy to filter the images. The policy
            should take the coco_eval object as input and return a list of
            (img_id, gt_ids, dt_ids) pairs. Both gt_ids and dt_ids are lists
            which records the gt/dt ids belong to the same image.
    """
    coco_dt: COCO = coco_eval.cocoDt
    coco_gt: COCO = coco_eval.cocoGt
    if filter_policy is None:
        filter_policy = default_filter_policy
    pairs = filter_policy(coco_eval)
    logger.info(f"Try to visualize {len(pairs)} items to {output_root}")
    pbar = tqdm(pairs)
    for img_id, gt_ids, dt_ids in pbar:
        pbar.set_description(f"Showing {img_id}")
        img_info = coco_gt.loadImgs([img_id])[0]
        image_path = os.path.join(image_root, img_info['file_name'])
        image = read_image(image_path)
        gt_anns = coco_gt.loadAnns(gt_ids)
        dt_anns = coco_dt.loadAnns(dt_ids)
        if len(gt_anns) == 0 and len(dt_anns) == 0:
            continue
        for ann in gt_anns:
            bbox = ann['bbox']
            image = plot_xywh_bbox(image, bbox)
        for ann in dt_anns:
            bbox = ann['bbox']
            image = plot_xywh_bbox(image, bbox, color=(255, 0, 0))
        output_path = os.path.join(output_root, img_info['file_name'])
        save_image(output_path, image)
    pbar.close()
