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




## 综述
| 标题 | 论文链接 | 发表位置 | 发表年份 |
| :---: | :---: | :---: | :---: |
| Segment anything model for medical image segmentation:Current  applications and future directions    | [paper](https://www.sciencedirect.com/science/article/pii/S0010482524003226) | Computers in Biology and Medicine   | 2024 |
| SAM2 for Image and Video Segmentation:A  Comprehensive Survey   | [paper](https://arxiv.org/pdf/2503.12781) | arxiv | 2025 |
|  |  |  |  |








## SAM-based
| 方法 | 标题 | 论文链接 | 代码链接 | 发表位置 | 核心思想 | 发表年份 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  SAM-PD |  SAM-PD: How Far Can SAM Take Us in Tracking and Segmenting Anything in Videos by Prompt Denoising   | [paper](https://arxiv.org/abs/2403.04194) | [code](https://github.com/infZhou/SAM-PD) | arxiv | 从前一帧的分割结果中提取点作为后一帧的prompt输入到SAM中 | 2024 |
| SAM2Long | SAM2LONG: ENHANCING SAM 2 FOR LONG VIDEO SEGMENTATION WITH A TRAINING-FREE MEM ORY TREE   | [paper](https://arxiv.org/abs/2410.16268) | [code](https://github.com/Mark12Ding/SAM2Long) | arxiv | 调整memory选择机制，防止选择错误进行累积 | 2024 |
| Medical SAM2 | <font style="color:rgb(31, 35, 40);">Medical SAM 2: Segment Medical Images As Video Via Segment Anything Model 2</font> | [paper](https://arxiv.org/abs/2408.00874) | [code](https://github.com/SuperMedIntel/Medical-SAM2) | arxiv | 提出了self-sorting memory bank | 2024 |
| SAM-I2V |  SAM-I2V: Upgrading SAM to Support Promptable Video Segmentation with Less than 0.2% Training Cost   | [paper](https://openaccess.thecvf.com/content/CVPR2025/html/Mei_SAM-I2V_Upgrading_SAM_to_Support_Promptable_Video_Segmentation_with_Less_CVPR_2025_paper.html) | [code](https://github.com/showlab/SAM-I2V) | CVPR | 利用图像和视频蕴含的时序关系得到prompt和原始prompt一起进行分割 | 2025 |
| SAM-PM | <font style="color:rgb(0, 0, 0);">SAM-PM: Enhancing Video Camouflaged Object Detection using Spatio-Temporal Attention</font> | [paper](https://openaccess.thecvf.com/content/CVPR2024W/PVUW/html/Meeran_SAM-PM_Enhancing_Video_Camouflaged_Object_Detection_using_Spatio-Temporal_Attention_CVPRW_2024_paper.html) | [code](https://github.com/SpiderNitt/SAM-PM) | CVPR | 加入空间时序交叉注意力机制模块来增强SAM的时序信息利用 | 2024 |
| <font style="color:rgb(51, 51, 51);">EP-SAM</font> | <font style="color:rgb(51, 51, 51);">EP-SAM: An Edge-Detection Prompt SAM Based Efficient Framework for Ultra-Low Light Video Segmentation</font> | [paper](https://ieeexplore.ieee.org/abstract/document/10889601) | [code](https://github.com/wzt22thu/EP-SAM/releases/tag/DEMO) | ICASSP | 使用轻量级边缘检测模型检测出图像物体边缘，作为prompt输入SAM进行分割 | 2025 |
| TSAM | <font style="color:rgb(0, 0, 0);">TSAM: Temporal SAM Augmented with Multimodal Prompts for Referring Audio-Visual Segmentation</font> | [paper](https://openaccess.thecvf.com/content/CVPR2025/html/Radman_TSAM_Temporal_SAM_Augmented_with_Multimodal_Prompts_for_Referring_Audio-Visual_CVPR_2025_paper.html) | 无 | CVPR | 融合图像、文本和音频三个模态生成prompt输入SAM进行分割 | 2025 |
| 无 |  SAM-AssistedTemporal-LocationEnhancedTransformerSegmentationfor Object TrackingwithOnlineMotionInference   | [paper](https://www.sciencedirect.com/science/article/pii/S0925231224016850) | 无 | Neurocomputing   | 利用SAM进行目标追踪，使用position prompt | 2025 |
| SAMUS | SAMUS:Adapting Segment Anything Model for Clinically-Friendly and Generalizable Ultrasound Image Segmentation   | [paper](https://arxiv.org/pdf/2309.06824v1) | [code](https://github.com/xianlin7/SAMUS) | arxiv | 加入CNN提取局部特征使用SAM提取全局特征进行融合 | 2023 |
| CC-SAM |  CC-SAM: SAM with Cross-feature Attention and Context for Ultrasound Image Segmentation   | [paper](https://arxiv.org/pdf/2408.00181) | 无 | ECCV | 在SAMUS的基础上使用chatgpt生成描述文本作为prompt | 2024 |
| SAMWISE | SAMWISE:Infusing Wisdom in SAM2 for Text-Driven Video Segmentation   | [paper](https://arxiv.org/pdf/2411.17646) | [code](https://github.com/ClaudiaCuttano/SAMWISE) | CVPR | 在SAM2的基础上加上了text prompt | 2025 |
| yolo-sam2 | <font style="color:rgb(0, 0, 0);">Self-Prompting Polyp Segmentation in Colonoscopy using Hybrid Yolo-SAM 2 Model</font> | [paper](https://export.arxiv.org/pdf/2409.09484) | [code](https://github.com/sajjad-sh33/YOLO_SAM2) | arxiv | yolo得到的检测框作为prompt | 2024 |
| <font style="color:rgb(0, 0, 0);">Det-SAM2</font> | <font style="color:rgb(0, 0, 0);">Det-SAM2:Technical Report on the Self-Prompting Segmentation Framework Based on Segment Anything Model 2</font> | [paper](https://arxiv.org/pdf/2411.18977) | [code](https://github.com/motern88/Det-SAM2) | arxiv | 提出了一个长视频处理SAM2框架 | 2024 |
| <font style="color:rgb(0, 0, 0);">SAMURAI</font> | <font style="color:rgb(0, 0, 0);">SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory</font> | [paper](https://arxiv.org/abs/2411.11922) | [code](https://github.com/yangchris11/samurai) | arxiv | 在memory机制上进行了创新 | 2024 |
| <font style="color:rgb(0, 0, 0);"> SurgSAM2  </font> | <font style="color:rgb(0, 0, 0);"> Surgical SAM 2: Real-time Segment Anything in Surgical Video by Efficient Frame Pruning  </font> | [paper](https://arxiv.org/pdf/2408.07931) | [code](https://github.com/jinlab-imvr/Surgical-SAM-2) | <font style="color:rgb(89, 99, 110);">NeurIPS</font> | memory中只保留包含信息的部分 | 2024 |
| <font style="color:rgb(0, 0, 0);">SurgicalSAM</font> | <font style="color:rgb(0, 0, 0);">SurgicalSAM: Efficient Class Promptable Surgical Instrument Segmentation</font> | [paper](https://arxiv.org/pdf/2308.08746) | code | <font style="color:rgb(89, 99, 110);">AAAI</font> | 使用手术器械的类别作为prompt输入到SAM | 2024 |
| <font style="color:rgb(0, 0, 0);">RMP-SAM</font> | <font style="color:rgb(0, 0, 0);">RMP-SAM: Towards Real-Time Multi-Purpose Segment Anything</font> | [paper](https://arxiv.org/abs/2401.10228) | [code](https://github.com/xushilin1/RMP-SAM) | <font style="color:rgb(89, 99, 110);">ICLR</font> | 生成prompt queries，使用动态卷积进行分割 | 2025 |




## 特征融合
| 方法 | 标题 | 论文链接 | 代码链接 | 发表位置 | 核心思想 | 发表年份 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| <font style="color:rgb(0, 0, 0);">无</font> | <font style="color:rgb(0, 0, 0);"> Dynamic Multimodal Fusion  </font> | [paper](https://openaccess.thecvf.com/content/CVPR2023W/MULA/papers/Xue_Dynamic_Multimodal_Fusion_CVPRW_2023_paper.pdf) | [code](https://github.com/zihuixue/DynMM) | <font style="color:rgb(89, 99, 110);">CVPR</font> | <font style="color:rgb(0, 0, 0);">利用门控机制和MOE进行多模态融合</font> | <font style="color:rgb(0, 0, 0);">2023</font> |

