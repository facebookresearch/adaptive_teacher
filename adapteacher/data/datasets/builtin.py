# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import contextlib
from detectron2.data import DatasetCatalog, MetadataCatalog
from fvcore.common.timer import Timer
# from fvcore.common.file_io import PathManager
from iopath.common.file_io import PathManager

from detectron2.data.datasets.pascal_voc import register_pascal_voc
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from .cityscapes_foggy import load_cityscapes_instances
import io
import logging

logger = logging.getLogger(__name__)

JSON_ANNOTATIONS_DIR = ""
_SPLITS_COCO_FORMAT = {}
_SPLITS_COCO_FORMAT["coco"] = {
    "coco_2017_unlabel": (
        "coco/unlabeled2017",
        "coco/annotations/image_info_unlabeled2017.json",
    ),
    "coco_2017_for_voc20": (
        "coco",
        "coco/annotations/google/instances_unlabeledtrainval20class.json",
    ),
}


def register_coco_unlabel(root):
    for _, splits_per_dataset in _SPLITS_COCO_FORMAT.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            meta = {}
            register_coco_unlabel_instances(
                key, meta, os.path.join(root, json_file), os.path.join(root, image_root)
            )


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
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info(
            "Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds())
        )

    id_map = None
    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())

    imgs = coco_api.loadImgs(img_ids)

    logger.info("Loaded {} images in COCO format from {}".format(len(imgs), json_file))

    dataset_dicts = []

    for img_dict in imgs:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        dataset_dicts.append(record)

    return dataset_dicts


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_coco_unlabel(_root)


# ==== Predefined splits for raw cityscapes foggy images ===========
_RAW_CITYSCAPES_SPLITS = {
    # "cityscapes_foggy_{task}_train": ("cityscape_foggy/leftImg8bit/train/", "cityscape_foggy/gtFine/train/"),
    # "cityscapes_foggy_{task}_val": ("cityscape_foggy/leftImg8bit/val/", "cityscape_foggy/gtFine/val/"),
    # "cityscapes_foggy_{task}_test": ("cityscape_foggy/leftImg8bit/test/", "cityscape_foggy/gtFine/test/"),
    "cityscapes_foggy_train": ("cityscapes_foggy/leftImg8bit/train/", "cityscapes_foggy/gtFine/train/"),
    "cityscapes_foggy_val": ("cityscapes_foggy/leftImg8bit/val/", "cityscapes_foggy/gtFine/val/"),
    "cityscapes_foggy_test": ("cityscapes_foggy/leftImg8bit/test/", "cityscapes_foggy/gtFine/test/"),
}


def register_all_cityscapes_foggy(root):
    # root = "manifold://mobile_vision_dataset/tree/yujheli/dataset"
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
def register_all_clipart(root):
    # root = "manifold://mobile_vision_dataset/tree/yujheli/dataset"
    SPLITS = [
        ("Clipart1k_train", "clipart", "train"),
        ("Clipart1k_test", "clipart", "test"),
    ]
    for name, dirname, split in SPLITS:
        year = 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"
        # MetadataCatalog.get(name).evaluator_type = "coco"

# ==== Predefined splits for Watercolor (PASCAL VOC format) ===========
def register_all_water(root):
    # root = "manifold://mobile_vision_dataset/tree/yujheli/dataset"
    SPLITS = [
        ("Watercolor_train", "watercolor", "train"),
        ("Watercolor_test", "watercolor", "test"),
    ]
    for name, dirname, split in SPLITS:
        year = 2012
        # register_pascal_voc(name, os.path.join(root, dirname), split, year, class_names=["person", "dog","bicycle", "bird", "car", "cat"])
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc_water"
        # MetadataCatalog.get(name).thing_classes = ["person", "dog","bike", "bird", "car", "cat"]
        # MetadataCatalog.get(name).thing_classes = ["person", "dog","bicycle", "bird", "car", "cat"]
        # MetadataCatalog.get(name).evaluator_type = "coco"

register_all_cityscapes_foggy(_root)
register_all_clipart(_root)
register_all_water(_root)

