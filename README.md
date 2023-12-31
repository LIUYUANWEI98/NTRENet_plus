# NTRENet++: Unleashing the Power of Non-target Knowledge for Few-shot Semantic Segmentation

This repo contains the code for our paper "*NTRENet++: Unleashing the Power of Non-target Knowledge for Few-shot Semantic Segmentation*" by Nian Liu, Yuanwei Liu, Yi Wu, Hisham Cholakkal, Rao Muhammad Anwer, Xiwen Yao, and Junwei Han. 

> **Abstract:** *Few-shot semantic segmentation (FSS) aims to segment the target object under the condition of a few annotated samples. However, current studies on FSS primarily concentrate on extracting information related to the  object, resulting in inadequate identification of ambiguous regions, particularly in non-target areas, including the background (BG) and Distracting Objects (DOs). Intuitively, to alleviate this problem, we propose a novel framework, namely NTRENet++, to explicitly mine and eliminate BG and DO regions in the query. First, we introduce a BG Mining Module (BGMM) to extract BG information and generate a comprehensive BG prototype from all images. For this purpose, a BG mining loss is formulated to supervise the learning of BGMM, utilizing only the known target object segmentation ground truth. Subsequently, based on this BG prototype, we employ a BG Eliminating Module to filter out the BG information from the query and obtain a BG-free result. Following this, the target information is utilized in the target matching module to generate the initial segmentation result. Finally, a DO Eliminating Module is proposed to further mine and eliminate DO regions, based on which we can obtain a BG and DO-free target object segmentation result. Moreover, we present a prototypical-pixel contrastive learning algorithm to enhance the model's capability to differentiate the target object from DOs. Extensive experiments conducted on both PASCAL-5i and COCO-20i datasets demonstrate the effectiveness of our approach despite its simplicity. Additionally, we extend our approach to the few-shot video segmentation task and achieve state-of-the-art performance on the YouTube-VIS dataset, demonstrating its generalization ability.*

<p align="middle">
  <img src="NTRENet++/img/framework1.jpg">
</p>
The framework of NTRENet++.

## :sparkles: **We extend the method on the task of FSVOS.**
<p align="middle">
  <img src="clip-NTRENet++/img/framwork2.jpg">
</p>
The framework of clip-NTRENet++.

## &#x1F527; Usage
### Dependencies

- Python 3.8
- PyTorch 1.7.0
- cuda 11.0
- torchvision 0.8.1
- tensorboardX 2.14

### Datasets and Data Preparation

Please download the following datasets:

+ PASCAL-5i is based on the [**PASCAL VOC 2012**](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) and [**SBD**](http://home.bharathh.info/pubs/codes/SBD/download.html) where the val images should be excluded from the list of training samples.

+ [**COCO 2014**](https://cocodataset.org/#download).

This code reads data from .txt files where each line contains the paths for image and the correcponding label respectively. Image and label paths are seperated by a space. Example is as follows:

    image_path_1 label_path_1
    image_path_2 label_path_2
    image_path_3 label_path_3
    ...
    image_path_n label_path_n

Then update the train/val/test list paths in the config files.

### Train with our Models
+ Update the config file by speficifying the target **split** and **path** (`weights`) for loading the checkpoint.
+ Execute `mkdir initmodel` at the root directory.
+ Download the ImageNet pretrained [**backbones**](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155122171_link_cuhk_edu_hk/EQEY0JxITwVHisdVzusEqNUBNsf1CT8MsALdahUhaHrhlw?e=4%3a2o3XTL&at=9) and put them into the `initmodel` directory.
+ Execute this command at the root directory:

    python train.py 
  
### Performance

Performance comparison with the state-of-the-art approaches (*i.e.*,  [PFENet](https://github.com/dvlab-research/PFENet)) in terms of **average** **mIoU** across all folds. 

##### PASCAL-5<sup>i</sup>

   | Backbone | Method     | 1-shot                   | 5-shot                   |
   | -------- | ---------- | ------------------------ | ------------------------ |
   | ResNet50    | PFENet      | 60.8                    | 61.9                    |
   |          | NTRENet++ (ours) | 65.3 <sub>(+4.5)</sub> | 66.4  <sub>(+4.5)</sub> |
   | ResNet101 | PFENet     | 60.1                   | 61.4                    |
   |          | NTRENet++ (ours) | 64.8 <sub>(+4.7)</sub> | 69.0 <sub>(+7.6)</sub> |

### Visualization

<p align="middle">
    <img src="NTRENet++/img/comparvis.jpg">
</p>

## References

This repo is mainly built based on [PFENet](https://github.com/dvlab-research/PFENet), [RePRI](https://github.com/mboudiaf/RePRI-for-Few-Shot-Segmentation), and [SemSeg](https://github.com/hszhao/semseg). Thanks for their great work!

## BibTeX

If you find our work and this repository useful. Please consider giving a star :star: and citation &#x1F4DA;.
