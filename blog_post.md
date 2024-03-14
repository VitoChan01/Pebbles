---
title: '
'
author: "Man Tuen Chan"
author_profile: true
date: 2024-03-12
# permalink:
toc: true
toc_sticky: true
toc_label: "
"
header:
  overlay_image: 
  overlay_filter: 0.3
  caption: "
  "
read_time: false
tags:
  - Stereo vision
  - segmentation
  - classification
  - U-Net
---

# Introduction
General idea:\
do a instance segmentation of pebbles see wt works better
why? search later

Sediment charateristics and grain-size distribution carries important information on the drainage system, the ecosystem, and the weather condition (Soloy et al., 2020; Wang et al., 2019). Unfortunately, traditional methods to manually collect these information are costly, labor intensive, and time consuming. Over the years, various techniques have been developed seeking to reduce manual input through machine learning. This project is an attempt to utilize a U-Net-SAM 2-pass approach in conjunction with stereo depth estimation to automize image based sampling. 

*This internship was supervised by Prof. Dr. Bodo Bookhagen.*

# Method
## Approach
This project approached the challenge from 2 angles. First, introduce a depth layer estimated through stereo disparity in addition to RGB. The rationale is to provide additional morphological information to improve the segmentatino accuracy. The use of stereo vision also provides the additional benefit that accurate size measurements is readily available and segmented results can easily be applied to the point cloud through the disparity map (Mustafah et al., 2012). 

Second, combine instance segmentation (Meta Segment-Anything) and semantic segmentation (U-Net) through a 2-pass approach to achieve extraction of individual grain samples. U-Net is a powerful tool capable of picking up features at different scale. Yet, while U-Net is capable of producing reliable results, it only performs a semantic segmentation. To separately sample individual grain, a second pass instance segmentation can be performed on top of the first pass U-Net result. The Meta Segment-Anything Model(SAM) was used in this project to perform the second pass segmentation. SAM is a pretrained model developed by Meta Platforms, Inc. (formerly Facebook). SAM was trained with large amount of data (1 billion masks and 11 million images) and evaluation of the model have shown that it provides reliable zero-shot performance (Kirillov et al., 2023). Should it performs reliably without the need of fine-tuning, significant time and computation power can be saved.

## Data collection
The Stereolabs ZED 2i stereo camera together with their software ZED SDK 4.0 was used to collect the data used in this project. The pebble setup (figure 1) was used as study object to collect in total 25 images. These 25 images were taken in 5 different angle, namely 1 (near-)nadir view and 4 top-down oblique view (backward, forward, left, and right). Each top-down oblique view were taken in 3 different height, giviing a high angle view image, mid angle view image, and a low angle view image. In total that adds up to 12 oblique images. While all remaining 13 images were take in nadir view, each nadir image were taken in different lighting condition. That includes variation in lighting intensity, direction, and number light source. 

<center>
<figure>
<a href=https://github.com/VitoChan01/Pebbles/blob/master/figure/example_setup.png?raw=true><img src=https://github.com/VitoChan01/Pebbles/blob/master/figure/example_setup.png?raw=true width="90%" height="90%" ></a><figcaption> Figure 1: Example of the pebble setup taken from middle-left angle. </figcaption>
</figure>
</center>

ZED 2i was used to retreive the RGB image, the depth estimation and the reprojected point cloud. Depth sensing was performed using the neural depth mode. This mode utilize model trained by Stereolabs to fill gaps and correct for noise. The estimated depth map was aligned to the image captured by the left sensor and comes in the HD2K resolution (2208x1242). For all 25 images, each image and measurement were average of 30 exposures to provide consistency and hand labeled using Napari.


## Model
Meta has provides 3 pretrained checkpoints for SAM. Three model have different neural network size, base ViT-B, large ViT-L, and huge ViT-H. ViT-H was used for the automatic mask generation with default settings.

The U-Net on the other hand was compiled using Adam algorithm with learning rate &alpha; = 0.1 for optimization. Binary cross-entropy loss function using binary IoU was chosen for binary segmentation, threshold was set at 0.5. The training was performed using batch size B = 64 and epoch e = 100. In total 4 models were trained. 2 models were trained using 13 nadir images. 13 images were divided into 4500 128x128 patches, of which ~20% were retained for validation. The remaining 12 oblique images were used as test images to evalue the model. They were not used in the training process. 2 models were trained using this dataset split, one with the depth layer and one without. For convenience, they were named split1-D (with depth) and split1-C (RGB only) respectively. The remaining 2 models were also differentiated by the use of depth layer but they were trained with a different dataset split. For these 2 models 9 nadir images and 4 oblique images (2 high angle, 1 mid angle, and 1 low angle) were used in the training instead of 13 nadir images. Following the same convention, they were named split2-D and split2-C.

The reason of using 2 different split has to do with a concern regarding the dataset. While variations were made in the view angle and lighting conditions, all images were essentially capturing the same study object. To examine if the training results can actually be generalized, split 1 and 2 restrict the view angles available to the training process as an attempt to evaluate the actual performance. For all models, an 8 fold kfold cross-validation was also performed.

# Results
## SAM segmentation
<center>
<figure>
<a href=https://github.com/VitoChan01/Pebbles/blob/master/figure/ZED_sam.png?raw=true><img src=https://github.com/VitoChan01/Pebbles/blob/master/figure/ZED_sam.png?raw=true width="80%" height="80%" ></a><figcaption> Figure 2: Segment-Anything segmentation of the pebble setup. </figcaption>
</figure>
</center>

<div style="display: flex; flex-direction: row; justify-content: center;">
    <figure style="margin-right: 20px;">
        <a href=https://github.com/VitoChan01/Pebbles/blob/master/figure/example_setup.png?raw=true><img src=https://github.com/VitoChan01/Pebbles/blob/master/figure/example_setup.png?raw=true width="90%" height="90%"></a>
        <figcaption>Figure 1: Example of the pebble setup taken from middle-left angle.</figcaption>
    </figure>
    <figure>
        <a href=https://github.com/VitoChan01/Pebbles/blob/master/figure/example_setup.png?raw=true><img src=https://github.com/VitoChan01/Pebbles/blob/master/figure/example_setup.png?raw=true width="90%" height="90%"></a>
        <figcaption>Figure 2: Another example.</figcaption>
    </figure>
</div>

## U-Net training history
<div style="display: flex; flex-direction: row; justify-content: center;">
    <figure style="margin-right: 20px;">
        <a href=https://github.com/VitoChan01/Pebbles/blob/master/figure/example_setup.png?raw=true><img src=https://github.com/VitoChan01/Pebbles/blob/master/figure/example_setup.png?raw=true width="90%" height="90%"></a>
        <figcaption>Figure 1: Example of the pebble setup taken from middle-left angle.</figcaption>
    </figure>
    <figure>
        <a href=https://github.com/VitoChan01/Pebbles/blob/master/figure/example_setup.png?raw=true><img src=https://github.com/VitoChan01/Pebbles/blob/master/figure/example_setup.png?raw=true width="90%" height="90%"></a>
        <figcaption>Figure 2: Another example.</figcaption>
    </figure>
</div>

## Accuracy
%can do?
check accuracy of combined approach. give some measurement of extracted pebble size? or extract a pebble point cloud?
# Conclusion and discussion

# Reference
Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., Dollár, P., & Girshick, R. (2023). Segment Anything. 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 3992–4003. https://doi.org/10.1109/ICCV51070.2023.00371
Mustafah, Y. M., Noor, R., Hasbi, H., & Azma, A. W. (2012). Stereo vision images processing for real-time object distance and size measurements. 2012 International Conference on Computer and Communication Engineering (ICCCE), 659–663. https://doi.org/10.1109/ICCCE.2012.6271270
Soloy, A., Turki, I., Fournier, M., Costa, S., Peuziat, B., & Lecoq, N. (2020). A Deep Learning-Based Method for Quantifying and Mapping the Grain Size on Pebble Beaches. Remote Sensing, 12(21), Article 21. https://doi.org/10.3390/rs12213659\
Wang, C., Lin, X., & Chen, C. (2019). Gravel Image Auto-Segmentation Based on an Improved Normalized Cuts Algorithm. Journal of Applied Mathematics and Physics, 7(3), Article 3. https://doi.org/10.4236/jamp.2019.73044\
