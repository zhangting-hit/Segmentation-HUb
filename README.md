<!--
 * @Author: zhangting
 * @Date: 2025-06-21 21:25:29
 * @LastEditors: Do not edit
 * @LastEditTime: 2025-06-22 09:39:39
 * @FilePath: /zhangting/Segmentation-Hub/README.md
-->
# Segmentation-Hub
本仓库旨在整理和汇总图像与视频序列分割（Image & Video Sequence Segmentation）相关的经典与前沿研究，包括论文、开源代码、项目链接及复现结果。重点关注时空信息建模、多模态融合、语义/实例分割、医学影像分割等方向，覆盖静态图像分割与动态图像序列分割两大领域。

## 自然图像领域
### 图像分割
| 方法 | 标题 | 论文链接 | 代码链接 | 发表位置 | 监督范式 | 发表年份 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| FCN | Fully Convolutional Networks for Semantic Segmentation | [paper](https://arxiv.org/abs/1411.4038) | [code](https://github.com/shelhamer/fcn.berkeleyvision.org) | CVPR | 全监督 | 2015 |
| U-Net | U-Net: Convolutional Networks for Biomedical Image Segmentation | [paper](https://arxiv.org/abs/1505.04597) | [code](https://github.com/milesial/Pytorch-UNet) | MICCAI | 全监督 | 2015 |
| SegNet | SegNet: A Deep Convolutional Encoder-Decoder Architecture | [paper](https://arxiv.org/abs/1511.00561) | [code](https://github.com/alexgkendall/SegNet-Tutorial) | arXiv | 全监督 | 2015 |
| DeepLab v3+ | Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation | [paper](https://arxiv.org/abs/1802.02611) | [code](https://github.com/tensorflow/models/tree/master/research/deeplab) | ECCV | 全监督 | 2018 |
| PSPNet | Pyramid Scene Parsing Network | [paper](https://arxiv.org/abs/1612.01105) | [code](https://github.com/hszhao/PSPNet) | CVPR | 全监督 | 2017 |
| Mask R-CNN | Mask R-CNN | [paper](https://arxiv.org/abs/1703.06870) | [code](https://github.com/facebookresearch/detectron2) | ICCV | 全监督 | 2017 |
| HRNet | Deep High-Resolution Representation Learning for Visual Recognition | [paper](https://arxiv.org/abs/1908.07919) | [code](https://github.com/HRNet/HRNet-Semantic-Segmentation) | TPAMI | 全监督 | 2019 |
| SegFormer | SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers | [paper](https://arxiv.org/abs/2105.15203) | [code](https://github.com/NVlabs/SegFormer) | NeurIPS | 全监督 | 2021 |
| SAM | Segment Anything | [paper](https://arxiv.org/abs/2304.02643) | [code](https://github.com/facebookresearch/segment-anything) | CVPR | 半监督 / 零样本 | 2023 |




### 视频序列分割
| 方法 | 标题 | 论文链接 | 代码链接 | 发表位置 | 监督范式 | 发表年份 |
| :---: | --- | :---: | :---: | :---: | :---: | :---: |
| OSVOS | One-Shot Video Object Segmentation | [paper](https://arxiv.org/abs/1611.05198) | [code](https://github.com/kmaninis/OSVOS-PyTorch) | CVPR | 半监督（首帧标注） | 2017 |
| MaskTrack R-CNN | <font style="color:rgb(0, 0, 0);">Video Instance Segmentation</font> | [paper](https://arxiv.org/abs/1905.04804) |  | ICCV | 全监督 | 2019 |
| STM | <font style="color:rgb(31, 35, 40);">Video Object Segmentation using Space-Time Memory Networks</font> | [paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Oh_Video_Object_Segmentation_Using_Space-Time_Memory_Networks_ICCV_2019_paper.pdf) | [code](https://github.com/seoungwugoh/STM) | ICCV | 半监督 | 2019 |
| AOT | Associating Objects with Transformers for Video Object Segmentation | [paper](https://arxiv.org/abs/2106.02638) | [code](https://github.com/yoxu515/aot-benchmark) | NeurIPS | 半监督 | 2021 |
| STEm-Seg | STEm-Seg: Spatio-Temporal Embeddings for Instance Segmentation in Videos | [paper](https://arxiv.org/abs/2003.08429) | [code](https://github.com/sabarim/STEm-Seg) | ECCV | 全监督 | 2020 |
| SSTVOS | <font style="color:rgb(0, 0, 0);">SSTVOS: Sparse Spatiotemporal Transformers for Video Object Segmentation</font> | [paper](https://arxiv.org/abs/2101.08833) | [code](https://github.com/dukebw/SSTVOS) | CVPR | 半监督 | 2021 |
| ReferFormer |  Language as Queries for Referring Video Object Segmentation   | [paper](https://arxiv.org/abs/2201.00487) | [code](https://github.com/wjn922/ReferFormer) | ECCV | 多模态监督（语言+掩码） | 2022 |
| XMem | Long-Term Video Object Segmentation with Memory Networks | [paper](https://arxiv.org/abs/2207.07115) | [code](https://github.com/hkchengrex/XMem) | ECCV | 半监督 | 2022 |




### 数据集
| 数据集名称 | 下载链接 | 简介 |
| :---: | :---: | :---: |
| PASCAL VOC | [PASCAL VOC官网](http://host.robots.ox.ac.uk/pascal/VOC/) | 经典的通用物体识别与分割数据集，包含20个类别的语义分割标签。 |
| MS COCO | [MS COCO官网](https://cocodataset.org/#download) | 大规模图像识别、检测与分割数据集，含丰富的实例分割标签。 |
| Cityscapes | [Cityscapes官网](https://www.cityscapes-dataset.com/downloads/) | 城市街景语义分割数据集，专注自动驾驶场景中的道路与物体分割。 |
| ADE20K | [ADE20K官网](http://groups.csail.mit.edu/vision/datasets/ADE20K/) | 多场景多类别图像分割数据集，覆盖室内外12000+张图像，150类别。 |
| CamVid | [CamVid下载链接](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) | 视频序列帧的像素级语义分割，专注动态交通场景。 |
| COCO-Stuff | [COCO-Stuff官网](https://github.com/nightrome/cocostuff) | MS COCO扩展，加入背景和材料类别的丰富语义标签。 |
| LVIS | [LVIS官网](https://www.lvisdataset.org/) | 大规模长尾实例分割数据集，涵盖1000+类别。 |
| KITTI Semantic | [KITTI官网](http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics) | 自动驾驶场景语义分割，含城市街景和车辆检测。 |
| Mapillary Vistas | [Mapillary官网](https://www.mapillary.com/dataset/vistas) | 城市街景数据集，丰富多样的户外语义分割标注。 |
| DAVIS | [DAVIS官网](https://davischallenge.org/) | 高质量视频对象分割数据集，常用于视频分割算法评测。 |




## 医学领域
### 图像分割(Image Segmentation)
| 方法 | 标题 | 论文链接 | 代码链接 | 发表位置 | 监督范式 | 发表年份 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| U-Net | U-Net: Convolutional Networks for Biomedical Image Segmentation | [paper](https://arxiv.org/abs/1505.04597) | [code](https://github.com/milesial/Pytorch-UNet) | MICCAI | 全监督 | 2015 |
| V-Net | V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation | [paper](https://arxiv.org/abs/1606.04797) | [code](https://github.com/faustomilletari/VNet) | 3DV | 全监督 | 2016 |
| Attention U-Net | Attention U-Net: Learning Where to Look for the Pancreas | [paper](https://arxiv.org/abs/1804.03999) | [code](https://github.com/hudyweas/brain-tumor-segmenator) | <font style="color:rgb(0, 0, 0);">MIDL</font> | 全监督 | 2018 |
| nnU-Net | nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation | [paper](https://arxiv.org/abs/1809.10486) | [code](https://github.com/MIC-DKFZ/nnUNet) | MICCAI | 全监督 | 2019 |
| TransUNet | TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation | [paper](https://arxiv.org/abs/2102.04306) | [code](https://github.com/Beckschen/TransUNet) | MIA | 全监督 | 2021 |
| UNETR | UNETR: Transformers for 3D Medical Image Segmentation | [paper](https://arxiv.org/abs/2103.10504) | [code](https://github.com/Project-MONAI/research-contributions/tree/main/UNETR) | <font style="color:rgb(0, 0, 0);">WACV</font> | 全监督 | 2022 |
| Swin-Unet | Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation | [paper](https://arxiv.org/abs/2105.05537) | [code](https://github.com/HuCaoFighting/Swin-Unet) | <font style="color:rgb(31, 35, 40);">ECCV</font> | 全监督 | 2022 |
| UNeXt | **<font style="color:rgb(31, 35, 40);">UNeXt</font>**<font style="color:rgb(31, 35, 40);">: MLP-based Rapid Medical Image Segmentation Network</font> | [paper](https://arxiv.org/pdf/2203.04967) | [code](https://github.com/jeya-maria-jose/UNeXt-pytorch) | <font style="color:rgb(0, 0, 0);">MICCAI</font> | 全监督 | 2022 |


### 视频序列分割
| 方法 | 标题 | 论文链接 | 代码链接 | 发表位置 | 监督范式 | 发表年份 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DCAN | Deep Contour-Aware Networks for Accurate Gland Segmentation | [paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Chen_DCAN_Deep_Contour-Aware_CVPR_2016_paper.pdf) |  | CVPR | 全监督 | 2016 |
|  RVOS   |  RVOS: End-to-End Recurrent Network for Video Object Segmentation   | [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ventura_RVOS_End-To-End_Recurrent_Network_for_Video_Object_Segmentation_CVPR_2019_paper.pdf) | [code](https://github.com/imatge-upc/rvos) | CVPR | 全监督 | 2019 |
| ViViT-Seg | ViViT-Seg: Video Vision Transformer for Medical Video Segmentation | [paper](https://arxiv.org/pdf/2103.15691) | [code](https://github.com/google-research/scenic/tree/main/scenic/projects/vivit) | ICCV | 全监督 | 2021 |
| PNSNet | <font style="color:rgb(0, 0, 0);">Progressively Normalized Self-Attention Network for Video Polyp Segmentation</font> | [paper](https://arxiv.org/abs/2105.08468) | [code](https://github.com/GewelsJI/PNS-Net) | MICCAI | 全监督 | 2021 |
| FLA-Net | <font style="color:rgb(0, 0, 0);">Shifting More Attention to Breast Lesion Segmentation in Ultrasound Videos</font> | [paper](https://arxiv.org/abs/2310.01861) | [code](https://github.com/jhl-Det/FLA-Net) | AAAI | 全监督 | 2022 |
| LGRNet | <font style="color:rgb(0, 0, 0);">HilbertMamba: Local-Global Reciprocal Network for Uterine Fibroid Segmentation in Ultrasound Videos</font> | [paper](https://arxiv.org/abs/2407.05703) | [code](https://github.com/bio-mlhui/LGRNet) | MICCAI | 全监督 | 2024 |
| MED-VT++ | <font style="color:rgb(0, 0, 0);">MED-VT++: Unifying Multimodal Learning with a Multiscale Encoder-Decoder Video Transformer</font> | [paper](https://arxiv.org/abs/2304.05930) | [code](https://github.com/rkyuca/medvt) | CVPR | 全监督 | 2024 |




### 数据集(Dataset)
| 数据集名称 | 下载链接 | 简介 |
| :---: | :---: | :---: |
| BraTS (Brain Tumor Segmentation) | [BraTS官网](https://www.synapse.org/Synapse:syn27046444/wiki/616571) | 脑肿瘤MRI多模态分割，包含多种肿瘤子区域标注。 |
| ISIC (International Skin Imaging Collaboration) | [ISIC挑战赛](https://challenge.isic-archive.com/data/) | 皮肤病变（痣、黑色素瘤）图像及病灶分割数据。 |
| LUNA16 (LUng Nodule Analysis 2016) | [LUNA16官网](https://luna16.grand-challenge.org/) | 肺部结节CT扫描数据，主要用于肺结节检测和分割。 |
| KiTS (Kidney Tumor Segmentation) | [KiTS Challenge](https://kits19.grand-challenge.org/) | 肾脏肿瘤CT扫描数据，包含肾脏和肿瘤标签。 |
| PROMISE12 (Prostate MR Image Segmentation) | [PROMISE12官网](https://promise12.grand-challenge.org/) | 前列腺MRI分割任务，重点是前列腺器官轮廓。 |
| Synapse Multi-organ Segmentation | [Synapse官网](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) | 多器官CT图像分割，包括肝、肾、脾等多个器官。 |
| ACDC (Automated Cardiac Diagnosis Challenge) | [ACDC Challenge](https://acdc.creatis.insa-lyon.fr/) | 心脏MRI分割数据集，包括左心室、右心室和心肌分割。 |
| CHAOS (Combined Healthy Abdominal Organ Segmentation) | [CHAOS Challenge](https://chaos.grand-challenge.org/) | 腹部多模态MRI和CT多器官分割数据。 |
| DRIVE (Digital Retinal Images for Vessel Extraction) | [DRIVE Dataset](https://drive.grand-challenge.org/) | 视网膜血管分割数据集，常用于眼科图像分析。 |
| AMOS (Abdominal Multi-Organ Segmentation) | [AMOS Challenge](https://amos22.grand-challenge.org/) | 包含多种腹部器官的CT图像和分割标签，适合多器官分割研究。 |
