# Copyright (c) 2020 by Jiachen (Jason) Zhou. All rights reserved.
#
# This file is used to train/fine-tune detectron2 Mask R-CNN on KITTI and Virtual KITTI 2 Datasets.
# Some lines are adopted from detectron2 https://github.com/facebookresearch/detectron2

import argparse
import glob
import os
import sys
import numpy as np
from fvcore.common.file_io import PathManager
from collections import OrderedDict
import torch
import detectron2
import detectron2.utils.comm as comm
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_cityscapes_instances, builtin_meta
from detectron2.engine import DefaultPredictor, DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import CityscapesInstanceEvaluator
from detectron2.utils.events import get_event_storage
from detectron2.utils.visualizer import Visualizer


def register_dataset_instance(image_dir, gt_dir, splits=["train", "val"], dataset_name="cityscapes", from_json=True):
    # use Cityscapes annotation format as metadata
    # CITYSCAPES_THING_CLASSES = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
    meta = builtin_meta._get_builtin_metadata("cityscapes")

    for split in splits:
        dataset_instance_name = str(dataset_name) + "_instance_" + str(split)
        # from_json = True if ground truth json annotation file is available
        DatasetCatalog.register(dataset_instance_name,
                                lambda: load_cityscapes_instances(image_dir+split, gt_dir+split,
                                                                  from_json=from_json, to_polygons=True))
        MetadataCatalog.get(dataset_instance_name).set(image_dir=image_dir+split, gt_dir=gt_dir+split,
                                                       evaluator_type="cityscapes_instance", **meta)
        print("finish registering dataset_{} to DatasetCatalog.".format(split))


class MyTrainer(DefaultTrainer):
    """Creat a subclass that inherits from detectron2's DefaultTrainer."""

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """Create an evaluator for cityscapes_instance evaluation."""
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "validation")
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        assert evaluator_type == "cityscapes_instance"
        assert (
                torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return MyCityscapesInstanceEvaluator(dataset_name)


class MyCityscapesInstanceEvaluator(CityscapesInstanceEvaluator):
    def evaluate(self):
        """
        Overwrite the evaluate method in CityscapesInstanceEvaluator.
        Add lines to write AP scores to be visualized in Tensorboard.
        """
        comm.synchronize()
        if comm.get_rank() > 0:
            return
        import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as cityscapes_eval

        self._logger.info("Evaluating results under {} ...".format(self._temp_dir))

        # set some global states in cityscapes evaluation API, before evaluating
        cityscapes_eval.args.predictionPath = os.path.abspath(self._temp_dir)
        cityscapes_eval.args.predictionWalk = None
        cityscapes_eval.args.JSONOutput = False
        cityscapes_eval.args.colorized = False
        cityscapes_eval.args.gtInstancesFile = os.path.join(self._temp_dir, "gtInstances.json")

        # These lines are adopted from
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py # noqa
        gt_dir = PathManager.get_local_path(self._metadata.gt_dir)
        groundTruthImgList = glob.glob(os.path.join(gt_dir, "*", "*_gtFine_instanceIds.png"))
        assert len(
            groundTruthImgList
        ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
            cityscapes_eval.args.groundTruthSearch
        )
        predictionImgList = []
        for gt in groundTruthImgList:
            predictionImgList.append(cityscapes_eval.getPrediction(gt, cityscapes_eval.args))
        results = cityscapes_eval.evaluateImgLists(
            predictionImgList, groundTruthImgList, cityscapes_eval.args
        )["averages"]

        res = OrderedDict()
        res["segm"] = {"AP": results["allAp"] * 100, "AP50": results["allAp50%"] * 100}

        # write evaluation AP scores to Tensorboard
        storage = get_event_storage()
        storage.put_scalar("eval/AP", res["segm"]["AP"])
        storage.put_scalar("eval/AP50", res["segm"]["AP50"])

        self._working_dir.cleanup()
        return res


def setup(args):
    cfg = get_cfg()
    # get config file from detectron2 model zoo
    if args.backbone == "resnet-50":
        cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))
    elif args.backbone == "resnext-152":
        cfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"))
    else:
        cfg.merge_from_file(model_zoo.get_config_file(args.backbone))
    cfg.merge_from_list(args.opts)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    dataset_root_dir = args.dataset_dir
    dataset_name = dataset_root_dir.split('/')[-1]
    image_dir = dataset_root_dir + "/data_semantics/"
    gt_dir = dataset_root_dir + "/gtFine/"

    # Register Dataset
    splits = ["train", "val"] if args.do_eval else ["train"]
    # Set from_json=False because there is no gt json for KITTI
    register_dataset_instance(image_dir, gt_dir, splits=splits, dataset_name=dataset_name, from_json=False)
    dataset_train = dataset_name + "_instance_train"
    dataset_val = dataset_name + "_instance_val"

    # Specify and create output directory
    cfg.OUTPUT_DIR = "{}/output_{}".format(args.output_dir, args.backbone)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Use the registered dataset for training and validation
    cfg.DATASETS.TRAIN = (dataset_train,)
    cfg.DATASETS.TEST = (dataset_val,) if args.do_eval else ()

    cfg.TEST.EVAL_PERIOD = 600

    cfg.DATALOADER.NUM_WORKERS = 4
    if args.ckpt_path is None:
        # load pre-trained weights
        if args.backbone == "resnet-50":
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Cityscapes/mask_rcnn_R_50_FPN.yaml")
        elif args.backbone == "resnext-152":
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml")
    else:
        cfg.MODEL.WEIGHTS = args.ckpt_path

    # config training parameters
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.01
    cfg.SOLVER.MAX_ITER = 30000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # default 512

    # choose trainer
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    return trainer.train()


if __name__ == '__main__':
    parser = default_argument_parser()
    parser.add_argument('--dataset_dir', type=str, default='./dataset/kitti_semantics_cs')
    parser.add_argument('--output_dir', type=str, default='./mask_rcnn_output')
    parser.add_argument('--backbone', type=str, default='resnet-50')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--ckpt_path', type=str, default=None)
    args = parser.parse_args()

    # Run training
    print("Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
