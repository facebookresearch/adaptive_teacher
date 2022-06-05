#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import contextlib
import io
import logging
import os
import json

from detectron2.data import DatasetCatalog, MetadataCatalog
from d2go.data.utils import CallFuncWithJsonFile
from detectron2.utils.file_io import PathManager
from fvcore.common.timer import Timer

from detectron2.data.datasets.pascal_voc import register_pascal_voc
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from .cityscapes_foggy import load_cityscapes_instances

logger = logging.getLogger(__name__)

_SPLITS_COCO_FORMAT = {}
_SPLITS_COCO_FORMAT["coco"] = {
    "coco_2017_unlabel": (
        "memcache_manifold://mobile_vision_dataset/tree/coco_unlabel2017",
        "memcache_manifold://mobile_vision_dataset/tree/coco_unlabel2017/coco_jsons/image_info_unlabeled2017.json",
    ),
    "goi_v5_unlabel": (
        "memcache_manifold://portal_ai_data/tree/goi_v5/train",
        "memcache_manifold://mobile_vision_dataset/tree/goi/v5/coco_jsons/openimages_v5_train_unlabel.json",
    ),
}


def register_coco_unlabel():
    for _, splits_per_dataset in _SPLITS_COCO_FORMAT.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            meta = {}
            register_coco_unlabel_instances(key, meta, json_file, image_root)


def register_coco_unlabel_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root

    # 1. register a function which returns dicts
    DatasetCatalog.register(
        name, lambda: load_coco_unlabel_json(json_file, image_root, name)
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def load_coco_unlabel_json(
    json_file, image_root, dataset_name=None, extra_annotation_keys=None
):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info(
            "Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds())
        )

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)
    logger.info(
        "Loaded {} unlabeled images in COCO format from {}".format(len(imgs), json_file)
    )

    dataset_dicts = []
    for img_dict in imgs:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        record["image_id"] = img_dict["id"]

        dataset_dicts.append(record)

    return dataset_dicts


_UNLABELED_DATASETS = {
    # 1-2 people images extracted from UGC ig images or ig profiles using fetch_image flow
    "UGC_unlabel_ig_1M_20210514_1or2people": "manifold://pai_mobile/tree/datasets/semi_supervised/unlabeled_UGC/sweep_4m_20210514_20210515_1or2people.json",
    # hand non-UGC long range frames extracted from collected videos
    "hand_nonUGC_long_range_384K_20210521": "manifold://pai_mobile/tree/datasets/hand_unlabeled_nonUGC/long_range.json",
    # hand non-UGC short range images cropped from the annotated bounding boxes in long-range videos
    "hand_nonUGC_short_range_183K_20210521": "manifold://pai_mobile/tree/datasets/hand_unlabeled_nonUGC/short_range.json",
}

def load_json(json_file):
    """
    Simply load and return the json_file
    """
    with PathManager.open(json_file, "r") as f:
        json_data = json.load(f)
    return json_data

def register_unlabeled():
    """
    Register the unlabeled datasets
    The json_file needs to be in D2's format
    """
    for name, json_file in _UNLABELED_DATASETS.items():
        # 1. register a function which returns dicts
        DatasetCatalog.register(
            name,
            CallFuncWithJsonFile(
                func=load_json,
                json_file=json_file
            )
        )

        # 2. Optionally, add metadata about this dataset,
        # since they might be useful in evaluation, visualization or logging
        MetadataCatalog.get(name).set(
            json_file=json_file, image_root="", evaluator_type="coco"
        )


# ==== Predefined splits for raw cityscapes foggy images ===========
_RAW_CITYSCAPES_SPLITS = {
    # "cityscapes_foggy_{task}_train": ("cityscape_foggy/leftImg8bit/train/", "cityscape_foggy/gtFine/train/"),
    # "cityscapes_foggy_{task}_val": ("cityscape_foggy/leftImg8bit/val/", "cityscape_foggy/gtFine/val/"),
    # "cityscapes_foggy_{task}_test": ("cityscape_foggy/leftImg8bit/test/", "cityscape_foggy/gtFine/test/"),
    "cityscapes_foggy_train": ("cityscape_foggy/leftImg8bit/train/", "cityscape_foggy/gtFine/train/"),
    "cityscapes_foggy_val": ("cityscape_foggy/leftImg8bit/val/", "cityscape_foggy/gtFine/val/"),
    "cityscapes_foggy_test": ("cityscape_foggy/leftImg8bit/test/", "cityscape_foggy/gtFine/test/"),
}


def register_all_cityscapes_foggy():
    root = "manifold://mobile_vision_dataset/tree/yujheli/dataset"
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        # inst_key = key.format(task="instance_seg")
        inst_key = key
        # DatasetCatalog.register(
        #     inst_key,
        #     lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
        #         x, y, from_json=True, to_polygons=True
        #     ),
        # )
        DatasetCatalog.register(
            inst_key,
            lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
                x, y, from_json=False, to_polygons=False
            ),
        )
        # MetadataCatalog.get(inst_key).set(
        #     image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes_instance", **meta
        # )
        # MetadataCatalog.get(inst_key).set(
        #     image_dir=image_dir, gt_dir=gt_dir, evaluator_type="pascal_voc", **meta
        # )
        MetadataCatalog.get(inst_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="coco", **meta
        )

# ==== Predefined splits for Clipart (PASCAL VOC format) ===========
def register_all_clipart():
    root = "manifold://mobile_vision_dataset/tree/yujheli/dataset"
    SPLITS = [
        ("Clipart1k_train", "clipart", "train"),
        ("Clipart1k_test", "clipart", "test"),
    ]
    for name, dirname, split in SPLITS:
        year = 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"
        # MetadataCatalog.get(name).evaluator_type = "coco"

register_all_cityscapes_foggy()
register_all_clipart()

# register_coco_unlabel()
# register_unlabeled()

def register_all_water():
    root = "manifold://mobile_vision_dataset/tree/yujheli/dataset" #Need to modify to the correct folder containing the dataset.
    SPLITS = [
        ("Watercolor_train", "watercolor", "train"),
        ("Watercolor_test", "watercolor", "test"),
    ]
    for name, dirname, split in SPLITS:
        year = 2012
        # register_pascal_voc(name, os.path.join(root, dirname), split, year, class_names=["person", "dog","bicycle", "bird", "car", "cat"])
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc_water"

register_all_water()

def register_all_clipart_ws():
    root = "manifold://mobile_vision_dataset/tree/yujheli/dataset"
    SPLITS = [
        ("Clipart1k_train_w", "clipart", "train"),
        ("Clipart1k_test_w", "clipart", "test"),
    ]
    for name, dirname, split in SPLITS:
        year = 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc_water"
        # MetadataCatalog.get(name).evaluator_type = "coco"

register_all_clipart_ws()