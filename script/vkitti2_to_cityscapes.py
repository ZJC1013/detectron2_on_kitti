# Copyright (c) 2020 by Jiachen (Jason) Zhou. All rights reserved.
#
# This file converts the Virtual KITTI 2 dataset to Cityscapes format.
# Modification is based on the original kitti_to_cityscapes.py provided by KITTI.

import json
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy.misc as sp
from tqdm import tqdm

join = os.path.join

# Manually create a mapping from Virtual Kitti 2 color scheme to Cityscapes instance_id
# Virtual KITTI 2 color scheme is described on this page:
# https://europe.naverlabs.com/research/computer-vision-research-naver-labs-europe/proxy-virtual-worlds-vkitti-2/
VKITTI2_CLASS_COLORS_TO_CITYSCAPES = {
    (0  , 0  ,  0 ): 0,   # unlabeled
    (210, 0  , 200): 22,  # terrain
    (90 , 200, 255): 23,  # sky
    (0  , 199,  0 ): 21,  # tree/vegetation
    (140, 140, 140): 11,  # building
    (100, 60 , 100): 7,   # road
    (250, 100, 255): 14,  # guard rail
    (255, 255,  0 ): 20,  # traffic sign
    (200, 200,  0 ): 19,  # traffic light
    (255, 130,  0 ): 17,  # pole
    (160, 60,   60): 27,  # truck
    (255, 127,  80): 26,  # car
    (0  , 139, 139): 29,  # van/caravan
}


def vkitti2_to_cityscapes_instaces(semantic_img, instance_img):
    # print(semantic_img.max())
    # print(instance_img.max())
    semantic_rst = np.zeros(instance_img.shape, dtype=np.uint8)
    for i in range(semantic_img.shape[0]):
        for j in range(semantic_img.shape[1]):
            semantic_rst[i, j] = VKITTI2_CLASS_COLORS_TO_CITYSCAPES.get(tuple(semantic_img[i, j]), 0)

    instance_mask = (instance_img > 0)
    instance_rst = (semantic_rst*1000 + instance_img)*instance_mask + semantic_rst*(1-instance_mask)
    return semantic_rst, instance_rst


if __name__ == '__main__':
    for scene in ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']:
        train_dir = '../../../../vkitti2/vkitti_2.0.3_rgb/{scene}/clone/frames/rgb/Camera_0/'.format(scene=scene)
        im_output_dir = '../../../../vkitti2/vkitti2_semantics_cs/data_semantics/train/{scene}/'.format(scene=scene)
        gt_output_dir = '../../../../vkitti2/vkitti2_semantics_cs/gtFine/train/{scene}/'.format(scene=scene)

        rgbimage_folder_name = 'vkitti_2.0.3_rgb'
        semantic_folder_name = 'vkitti_2.0.3_classSegmentation'
        instance_folder_name = 'vkitti_2.0.3_instanceSegmentation'

        training_dir = join(train_dir)
        semantic_dir = train_dir.replace(rgbimage_folder_name,
                                         semantic_folder_name).replace('rgb', semantic_folder_name.split('_')[-1])
        instance_dir = train_dir.replace(rgbimage_folder_name,
                                         instance_folder_name).replace('rgb', instance_folder_name.split('_')[-1])
        out_semantic_dir = join(gt_output_dir)
        out_instance_dir = join(gt_output_dir)

        for d in [im_output_dir, out_semantic_dir, out_instance_dir]:
            if not os.path.exists(d):
                os.makedirs(d)

        training_file_list = [f for f in os.listdir(training_dir) if os.path.isfile(join(training_dir, f))]

        for f in tqdm(training_file_list):
            semantic_img = sp.imread(join(semantic_dir, f.replace('rgb', 'classgt').replace('jpg', 'png')))
            instance_img = sp.imread(join(instance_dir, f.replace('rgb', 'instancegt').replace('jpg', 'png')), mode='L')

            semantic_rst, instance_rst = vkitti2_to_cityscapes_instaces(semantic_img, instance_img)

            out_semantic_filename = join(out_semantic_dir, '{}_000000_{}_gtFine_labelIds.png'.format(scene, f[4:-4]))
            out_instance_filename = join(out_instance_dir, '{}_000000_{}_gtFine_instanceIds.png'.format(scene, f[4:-4]))
            out_polygons_filename = join(out_instance_dir, '{}_000000_{}_gtFine_polygons.json'.format(scene, f[4:-4]))

            sp.toimage(semantic_rst, mode='L').save(out_semantic_filename)
            sp.toimage(instance_rst, high=np.max(instance_rst), low=np.min(instance_rst), mode='I').save(out_instance_filename)

            # create empty json file for pseudo polygons
            with open(out_polygons_filename, 'w') as out_json:
                json.dump({}, out_json)

            # copy and rename kitti semantics training image_2 to cityscapes format
            training_img_src = join(training_dir, f)
            training_img_dst = join(im_output_dir, '{}_000000_{}_leftImg8bit.png'.format(scene, f[4:-4]))
            shutil.copy2(training_img_src, training_img_dst)
