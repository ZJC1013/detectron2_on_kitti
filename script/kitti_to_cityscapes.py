# Copyright (c) 2020 by Jiachen (Jason) Zhou. All rights reserved.
#
# This kitti_to_cityscapes format conversion file is modified based on
# the original file provided by KITTI, which is available here
# https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_semantics.zip

import json
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as sp
from tqdm import tqdm

join = os.path.join


def kitti_to_cityscapes_instaces(instance_img):
    kitti_semantic = instance_img // 256
    kitti_instance = instance_img % 256
    # print(kitti_semantic.max())
    # print(kitti_instance.max())

    instance_mask = (kitti_instance > 0)
    cs_instance = (kitti_semantic*1000 + kitti_instance)*instance_mask + kitti_semantic*(1-instance_mask)
    return cs_instance


if __name__ == '__main__':
    train_dir = '../../../data_semantics/training/'
    im_output_dir = '../../../kitti_semantics_cs/data_semantics/train/kitti/'
    gt_output_dir = '../../../kitti_semantics_cs/gtFine/train/kitti/'
    
    training_dir = join(train_dir, 'image_2/')
    semantic_dir = join(train_dir, 'semantic/')
    instance_dir = join(train_dir, 'instance/')
    out_semantic_dir = join(gt_output_dir)
    out_instance_dir = join(gt_output_dir)

    for d in [im_output_dir, out_semantic_dir, out_instance_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    semantic_file_list = [f for f in os.listdir(semantic_dir) if os.path.isfile(join(semantic_dir, f))]

    for f in tqdm(semantic_file_list):
        semantic_img = sp.imread(join(semantic_dir, f))
        instance_img = sp.imread(join(instance_dir, f))

        instance_img = kitti_to_cityscapes_instaces(instance_img)

        out_semantic_filename = join(out_semantic_dir, 'kitti_%s_gtFine_labelIds.png'%f[:-4])
        out_instance_filename = join(out_instance_dir, 'kitti_%s_gtFine_instanceIds.png'%f[:-4])
        out_polygons_filename = join(out_instance_dir, 'kitti_%s_gtFine_polygons.json'%f[:-4])

        sp.toimage(semantic_img, mode='L').save(out_semantic_filename)
        sp.toimage(instance_img, high=np.max(instance_img), low=np.min(instance_img), mode='I').save(out_instance_filename)

        # create empty json file for pseudo polygons
        with open(out_polygons_filename, 'w') as out_json:
            json.dump({}, out_json)

        # copy and rename kitti semantics training image_2 to cityscapes format
        training_img_src = join(training_dir, f)
        training_img_dst = join(im_output_dir, 'kitti_%s_leftImg8bit.png'%f[:-4])
        shutil.copy2(training_img_src, training_img_dst)
