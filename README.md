# Detectron2 Mask R-CNN Fine-tune on KITTI Semantics Dataset
## Getting Started
This release is implemented and tested on Ubuntu 16.04 with 4 GeForce GTX 1080 Ti GPU cards,
Python 3.6.9, PyTorch 1.6.0 and detectron2 0.2.1 .  

1. Clone this repo and also detectron2.  
    ```
    git clone https://github.com/ZJC1013/detectron2_on_kitti.git
    git clone https://github.com/facebookresearch/detectron2.git
    ```

2. Create a detectron2 docker  
See the official [instruction](https://github.com/facebookresearch/detectron2/blob/master/docker/README.md) 
3. Install additional packages inside the docker container
    ```
    cd detectron2_on_kitti
    pip install -r requirements.txt
    ```

## Training
### Dataset
To train/fine-tune on the 
[KITTI Semantics Dataset](http://www.cvlibs.net/datasets/kitti/eval_instance_seg.php?benchmark=instanceSeg2015) and 
[Virtual KITTI 2 Dataset](
https://europe.naverlabs.com/research/computer-vision-research-naver-labs-europe/proxy-virtual-worlds-vkitti-2/)
+ Download the data `data_semantics` and the official development kit `devkit_semantics` from KITTI
+ Download the data `vkitti_2.0.3_rgb.tar`, `vkitti_2.0.3_classSegmentation.tar`, and 
`vkitti_2.0.3_instanceSegmentation.tar` from Virtual KITTI 2. (Optional)
+ Replace the `devkit_semantics/devkit/helpers/kitti_to_cityscapes.py` by the one provided in this repo 
`./script/kitti_to_cityscapes.py`
+ Copy the `./script/vkitti2_to_cityscapes.py` to the same location with 
`devkit_semantics/devkit/helpers/kitti_to_cityscapes.py` (Optional)
Your folder should look like the following:  
```
/path/to/downloaded/data/
    kitti
        data_semantics
        devkit_semantics
            devkit
                helpers
                    kitti_to_cityscapes.py  # replaced
                    vkitti2_to_cityscapes.py  # new
    vkitti2
        vkitti_2.0.3_rgb
        vkitti_2.0.3_classSegmentation
        vkitti_2.0.3_instanceSegmentation
```
#### Dataset format conversion
Due to the fact that dectectron2 supports Cityscapes format, and KITTI semantics are created to conform with Cityscapes, 
though there are differences, we need to use scripts `kitti_to_cityscapes.py` and `vkitti2_to_cityscapes.py` to 
convert KITTI semantics data and Virual KITTI 2 data into Cityscapes format. *Failing to do so would result in not able 
to load and register the dataset to detectron2 `DatasetCatalog`.*  
```
# inside your docker container
cd /path/to/downloaded/data/kitti/devkit_semantics/devkit/helpers
python kitti_to_cityscapes.py  # should only take less than a minute to finish
python vkitti2_to_cityscapes.py  # should take about 3 hours to finish
```
After running these two conversion scripts, your folder structure will become:  
```
/path/to/downloaded/data/
    kitti
        data_semantics
        devkit_semantics
        kitti_semantics_cs  # newly generated, 'cs' stands for cityscapes
            data_semantics
                train
                    kitti  # contains 200 images
                        kitti_000000_10_leftImg8bit.png, ..., kitti_000199_10_leftImg8bit.png
            gtFine
                train
                    kitti  # contains 3 * 200 files
                        kitti_000000_10_gtFine_instanceIds.png
                        kitti_000000_10_gtFine_labelIds.png
                        kitti_000000_10_gtFine_polygons.json
                        ...
                        kitti_000199_10_gtFine_polygons.json
    vkitti2
        vkitti_2.0.3_rgb
        vkitti_2.0.3_classSegmentation
        vkitti_2.0.3_instanceSegmentation
        vkitti2_semantics_cs  # newly generated
            data_semantics
                train
                    Scene01  # contains 447 images
                        Scene01_000000_00000_leftImg8bit.png, ..., Scene01_000000_00446_leftImg8bit.png
                    ...
                    Scene20
            gtFine
                train
                    Scene01  # contains 3 * 447 images
                        Scene01_000000_00000_gtFine_instanceIds.png
                        Scene01_000000_00000_gtFine_labelIds.png
                        Scene01_000000_00000_gtFine_polygons.json
                        ...
                        Scene01_000000_00446_gtFine_polygons.json
                    ...
                    Scene20
```
For convenience, if you do not want to convert those datasets yourself, you may choose to download the converted 
`kitti_semantics_cs` and `vkitti2_semantics_cs` that I uploaded to a Google Drive with public link here: 
[KITTI_Semantic_in_Cityscapes_Format](
https://drive.google.com/drive/folders/13XmMwoorFfkHE8HTRywQp_RylBjRw1xx?usp=sharing)


### Run training
It is recommended that you create symbolic links in `./dataset/` to your actual dataset location.
```
cd detectron2_on_kitti
cd dataset
# create symlink
ln -sfn /path/to/downloaded/data/kitti/kitti_semantics_cs kitti_semantics_cs
```
+ To run training on KITTI without evaluation 
```
cd detectron2_on_kitti
python detectron2_mask_rcnn.py --num-gpus 4 --output_dir ./mask_rcnn_output --backbone resnet-50
```

+ To run training on KITTI with evaluation on Virtual KITTI 2 as val set  
You need to add val set symlinks to `kitti_semantics_cs`
```
cd detectron2_on_kitti/dataset
cd kitti_semantics_cs/data_semantics
ln -sfn /path/to/downloaded/data/vkitti2/vkitti2_semantics_cs/data_semantics/train val
cd ../gtFine
ln -sfn /path/to/downloaded/data/vkitti2/vkitti2_semantics_cs/gtFine/train val
```
```
cd detectron2_on_kitti
python detectron2_mask_rcnn.py --num-gpus 4 --output_dir ./mask_rcnn_output --backbone resnet-50  --do_eval
```
You may want to set a number for `cfg.TEST.EVAL_PERIOD` in main() which controls how many iterations apart an 
evaluation should perform. After each evaluation, two scores `AP` and `AP50` will be reported in both console output 
and in Tensorboard `eval` section.
