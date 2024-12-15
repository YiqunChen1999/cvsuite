import os
import numpy as np
import cv2
from PIL import Image
from torch import Tensor as T

from cvsuite.structures.ops import xywh_to_xyxy


def read_image(image_path: str) -> np.ndarray:
    """
    Read an image from the path.

    Args:
        image_path: The path of the image.

    Returns:
        The image.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_image(image_path: str, image: np.ndarray):
    """
    Save the image to the path.

    Args:
        image_path: The path to save the image.
        image: The image to save.
    """
    dirname = os.path.dirname(image_path)
    os.makedirs(dirname, exist_ok=True)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, image)


def plot_xywh_bbox(image: str | Image.Image | np.ndarray,
                   bbox: list[int | float] | np.ndarray,
                   color: tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Show the bounding box on the image.

    Args:
        image: The image to show.
        bbox: The bounding box to show.
    """
    if not isinstance(bbox, (np.ndarray, T)):
        bbox = np.asarray(bbox)
    xyxy = xywh_to_xyxy(bbox)
    return plot_xyxy_bbox(image, xyxy, color)


def plot_xyxy_bbox(image: str | Image.Image | np.ndarray,
                   bbox: list[int | float] | np.ndarray | T,
                   color: tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Show the bounding box on the image.

    Args:
        image: The image to show.
        bbox: The bounding box to show.
    """
    if isinstance(image, str):
        image = read_image(image)
    elif isinstance(image, Image.Image):
        image = np.array(image)
    x0, y0 = int(bbox[0]), int(bbox[1])
    x1, y1 = int(bbox[2]), int(bbox[3])
    image = cv2.rectangle(image, (x0, y0), (x1, y1), color, 1)
    return image
