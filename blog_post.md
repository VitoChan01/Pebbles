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
The Stereolabs ZED 2i stereo camera together with their software ZED SDK 4.0 was used to collect the data used in this project. The pebble setup (figure 1) was used as study object to collect in total 25 images. These 25 images were taken in 5 different angle, namely 1 (near-)nadir view and 4 top-down oblique view (backward, forward, left, and right). Each top-down oblique view were taken in 3 different height, giviing a high view image, mid view image, and a low view image. In total that adds up to 12 oblique images. While all remaining 13 images were take in nadir view, each nadir image were taken in different lighting condition. That includes variation in lighting intensity, direction, and number light source. 

<center>
<figure>
<a href=https://github.com/VitoChan01/Pebbles/blob/master/figure/example_setup.png?raw=true><img src=https://github.com/VitoChan01/Pebbles/blob/master/figure/example_setup.png?raw=true width="90%" height="90%" ></a><figcaption> Figure 1: Example of the pebble setup taken from middle-left angle. </figcaption>
</figure>
</center>

## Model construction

# Results
training history
# Accuracy
%can do?
check accuracy of combined approach. give some measurement of extracted pebble size? or extract a pebble point cloud?
# Conclusion and discussion

# Reference
Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., Dollár, P., & Girshick, R. (2023). Segment Anything. 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 3992–4003. https://doi.org/10.1109/ICCV51070.2023.00371
Mustafah, Y. M., Noor, R., Hasbi, H., & Azma, A. W. (2012). Stereo vision images processing for real-time object distance and size measurements. 2012 International Conference on Computer and Communication Engineering (ICCCE), 659–663. https://doi.org/10.1109/ICCCE.2012.6271270
Soloy, A., Turki, I., Fournier, M., Costa, S., Peuziat, B., & Lecoq, N. (2020). A Deep Learning-Based Method for Quantifying and Mapping the Grain Size on Pebble Beaches. Remote Sensing, 12(21), Article 21. https://doi.org/10.3390/rs12213659\
Wang, C., Lin, X., & Chen, C. (2019). Gravel Image Auto-Segmentation Based on an Improved Normalized Cuts Algorithm. Journal of Applied Mathematics and Physics, 7(3), Article 3. https://doi.org/10.4236/jamp.2019.73044\
