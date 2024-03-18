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
The Stereolabs ZED 2i stereo camera together with their software ZED SDK 4.0 was used to collect the data used in this project. The pebble setup (figure 1 and 2) was used as study object to collect in total 25 images. These 25 images were taken in 5 different angle, namely 1 (near-)nadir view and 4 top-down oblique view (backward, forward, left, and right). Each top-down oblique view were taken in 3 different angle, giviing a high angle view image, mid angle view image, and a low angle view image. In total that adds up to 12 oblique images. While all remaining 13 images were take in nadir view, each nadir image were taken in different lighting condition. That includes variation in lighting intensity, direction, and number light source. 


<div style="display: flex; flex-direction: row; justify-content: center;">
    <figure style="flex: 1; margin-right: 0px;">
        <a href=https://github.com/VitoChan01/Pebbles/blob/master/figure/high_imgL.jpg?raw=true><img src=https://github.com/VitoChan01/Pebbles/blob/master/figure/high_imgL.jpg?raw=true width="100%" height="100%"></a>
        <figcaption>a: Left image</figcaption>
    </figure>
    <figure style="flex: 1; margin-right: 0px;">
        <a href=https://github.com/VitoChan01/Pebbles/blob/master/figure/high_imgR.jpg?raw=true><img src=https://github.com/VitoChan01/Pebbles/blob/master/figure/high_imgR.jpg?raw=true width="100%" height="100%"></a>
        <figcaption>b: Right image</figcaption>
    </figure>
    <figure style="flex: 1; margin-right: 0px;">
        <a href=https://github.com/VitoChan01/Pebbles/blob/master/figure/high_depth.jpg?raw=true><img src=https://github.com/VitoChan01/Pebbles/blob/master/figure/high_depth.jpg?raw=true width="100%" height="100%"></a>
        <figcaption>c: Depth image</figcaption>
    </figure>
</div>
<br/>
<center>
    <figcaption>Figure 1: Example of the pebble setup taken from high-left angle.</figcaption>
</center>



<div style="display: flex; flex-direction: row; justify-content: center;">
    <figure style="flex: 1; margin-right: 0px;">
        <a href=https://github.com/VitoChan01/Pebbles/blob/master/figure/low_imgL.jpg?raw=true><img src=https://github.com/VitoChan01/Pebbles/blob/master/figure/low_imgL.jpg?raw=true width="100%" height="100%"></a>
        <figcaption>a: Left image</figcaption>
    </figure>
    <figure style="flex: 1; margin-right: 0px;">
        <a href=https://github.com/VitoChan01/Pebbles/blob/master/figure/low_imgR.jpg?raw=true><img src=https://github.com/VitoChan01/Pebbles/blob/master/figure/low_imgR.jpg?raw=true width="100%" height="100%"></a>
        <figcaption>b: Right image</figcaption>
    </figure>
    <figure style="flex: 1; margin-right: 0px;">
        <a href=https://github.com/VitoChan01/Pebbles/blob/master/figure/low_depth.jpg?raw=true><img src=https://github.com/VitoChan01/Pebbles/blob/master/figure/low_depth.jpg?raw=true width="100%" height="100%"></a>
        <figcaption>c: Depth image</figcaption>
    </figure>
</div>
<br/>
<center>
    <figcaption>Figure 2: Example of the pebble setup taken from low-left angle.</figcaption>
</center>


ZED 2i was used to retreive the RGB image, the depth estimation and the reprojected point cloud (figure 3). Depth sensing was performed using the neural depth mode and stored in meter unit. This mode utilize model trained by Stereolabs to fill gaps and correct for noise. The estimated depth map was aligned to the image captured by the left sensor and comes in the HD2K resolution (2208x1242). For all 25 images, each image and measurement were average of 30 exposures to provide consistency and hand labeled using Napari. As shown in figure 3, depth sensing can be quite challenging at steep slopes, overhangs, and edges where a drastic change in distance can be found. There were visble signs of artifacts that is possibly the result of the model's attempt to interpolate and smooth the transition. At a lower angle, shadowing can also be seen. Combined with the gap filling artifacts and the limitation of hardware resolution, the use of depth maps may actually hinder the segmentation instead of benefit it when it comes to lower angle shots. 

<div style="display: flex; flex-direction: row; justify-content: center;">
    <figure style="flex: 1; margin-right: 0px;">
        <a href=https://github.com/VitoChan01/Pebbles/blob/master/figure/high_point_cloud.png?raw=true><img src=https://github.com/VitoChan01/Pebbles/blob/master/figure/high_point_cloud.png?raw=true width="100%" height="100%"></a>
        <figcaption>a: High-left angle</figcaption>
    </figure>
    <figure style="flex: 1; margin-right: 0px;">
        <a href=https://github.com/VitoChan01/Pebbles/blob/master/figure/low_point_cloud.png?raw=true><img src=https://github.com/VitoChan01/Pebbles/blob/master/figure/low_point_cloud.png?raw=true width="100%" height="100%"></a>
        <figcaption>b: Low-left angle</figcaption>
    </figure>
</div>
<br/>
<center>
<figcaption>Figure 3: Example of the pebble point cloud generated.</figcaption>
</center>


## Model
Meta has provides 3 pretrained checkpoints for SAM. Three model have different neural network size, base ViT-B, large ViT-L, and huge ViT-H. ViT-H was used for the automatic mask generation with default settings.

The U-Net on the other hand was compiled using Adam algorithm with learning rate &alpha; = 0.1 for optimization. Binary cross-entropy loss function using binary IoU was chosen for binary segmentation, threshold was set at 0.5. The training was performed using batch size B = 64 and epoch e = 100. In total 4 models were trained. 2 models were trained using 13 nadir images. 13 images were divided into 4500 128x128 patches, of which ~20% were retained for validation. The remaining 12 oblique images were used as test images to evalue the model. They were not used in the training process. 2 models were trained using this dataset split, one with the depth layer and one without. For convenience, they were named split1D (with depth) and split1C (RGB only) respectively. The remaining 2 models were also differentiated by the use of depth layer but they were trained with a different dataset split. For these 2 models 9 nadir images and 4 oblique images (2 high angle, 1 mid angle, and 1 low angle) were used in the training instead of 13 nadir images. Following the same convention, they were named split2D and split2C.

The reason of using 2 different split has to do with a concern regarding the dataset. While variations were made in the view angle and lighting conditions, all images were essentially capturing the same study object. To examine if the training results can actually be generalized, split 1 and 2 restrict the view angles available to the training process as an attempt to evaluate the actual performance. 

For all models, an 8 fold kfold cross-validation was also performed. The training and validation dataset were shuffled before the split and all models uses the same split for all 8 folds. 

# Results
## SAM segmentation
The results of SAM segmentation were demonstrated below. SAM returns a binary mask for each individual segments, figure 4 shows an example of aggregated mask plotted with the original image. SAM model was trained with RGB photos hence depth image was not used. Figure 5 plots the accuracy of the segmentation by image. Image 0 to 12 were nadir images with variation in lighting condition, 13-24 were oblique image with variation in view angle. The accuracy was assessed by pixel and by pebble. By pixel consider if a pixel that was manually labeled 
%%write how ac were calculated

<center>
<figure>
<a href=https://github.com/VitoChan01/Pebbles/blob/master/figure/ZED_sam.png?raw=true><img src=https://github.com/VitoChan01/Pebbles/blob/master/figure/ZED_sam.png?raw=true width="80%" height="80%" ></a><figcaption> Figure 4: Segment-Anything segmentation of the pebble setup. </figcaption>
</figure>
</center>
<center>
<figure >
    <a href=https://github.com/VitoChan01/Pebbles/blob/master/figure/sam_pixel_count.png?raw=true><img src=https://github.com/VitoChan01/Pebbles/blob/master/figure/sam_pixel_count.png?raw=true width="50%" height="40%"></a>
    <figcaption>Figure 5: Accuracy of SAM segmentation.</figcaption>
</figure>
</center>


## U-Net training history and accuracy
Figure 6 shows the training history of all splits at the first fold. Accuracy accessed through precesion, accuracy, F1 score, and binary IoU. The metrics were only calculated for test images that the models have seen in the training process. Figure 7 shows the accuracy metrics by image at different threshold. Solid line shows the mean value over 8 folds kfold and the shaded area highlights the 25th to 75th percentile. For the view angle correponding to each test images please see table 1. Figure 8 shows the the accuracy variation over the folds. Solid line represent the mean value over all test images with threshold=0.7 while the shaded area indicates the threhold = 0.5-0.9.

<div style="display: flex; flex-direction: row; justify-content: center;">
    <figure style="flex: 1; margin-right: 0px;">
        <a href=https://github.com/VitoChan01/Pebbles/blob/master/figure/history/pebble_unet_Split1C_kfold0.png?raw=true><img src=https://github.com/VitoChan01/Pebbles/blob/master/figure/history/pebble_unet_Split1C_kfold0.png?raw=true width="100%" height="100%"></a>
        <figcaption>a: Split1C Kfold 0</figcaption>
    </figure>
    <figure style="flex: 1; margin-right: 0px;">
        <a href=https://github.com/VitoChan01/Pebbles/blob/master/figure/history/pebble_unet_Split1C_kfold0.png?raw=true><img src=https://github.com/VitoChan01/Pebbles/blob/master/figure/history/pebble_unet_Split1C_kfold0.png?raw=true width="100%" height="100%"></a>
        <figcaption>b: Split1D Kfold 0</figcaption>
    </figure>
</div>
<div style="display: flex; flex-direction: row; justify-content: center;">
    <figure style="flex: 1; margin-right: 0px;">
        <a href=https://github.com/VitoChan01/Pebbles/blob/master/figure/history/pebble_unet_Split1C_kfold0.png?raw=true><img src=https://github.com/VitoChan01/Pebbles/blob/master/figure/history/pebble_unet_Split1C_kfold0.png?raw=true width="100%" height="100%"></a>
        <figcaption>c: Split2C Kfold 0</figcaption>
    </figure>
    <figure style="flex: 1; margin-right: 0px;">
        <a href=https://github.com/VitoChan01/Pebbles/blob/master/figure/history/pebble_unet_Split1C_kfold0.png?raw=true><img src=https://github.com/VitoChan01/Pebbles/blob/master/figure/history/pebble_unet_Split1C_kfold0.png?raw=true width="100%" height="100%"></a>
        <figcaption>d: Split2D Kfold 0</figcaption>
    </figure>
</div>
<br/>
<center>
<figcaption>Figure 6: Model training history.</figcaption>
</center>

Table 1. Test images
<center>

| Split | Nadir | Oblique high | Oblique mid | Oblique low |
|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
| split1   | -- | Image 0-3 | Image 4-7 | Image 7-11 |
| split2   | Image 8-11 | Image 0-1| Image 2-4 | Image 5-7 |
</center>

<div style="display: flex; flex-direction: row; justify-content: center;">
    <figure style="flex: 1; margin-right: 0px;">
        <a href=https://github.com/VitoChan01/Pebbles/blob/master/figure/Accuracy_plot/pebble_unet_Split1C_image_accuracy.png?raw=true><img src=https://github.com/VitoChan01/Pebbles/blob/master/figure/Accuracy_plot/pebble_unet_Split1C_image_accuracy.png?raw=true width="100%" height="100%"></a>
        <figcaption>a: Split1C</figcaption>
    </figure>
    <figure style="flex: 1; margin-right: 0px;">
        <a href=https://github.com/VitoChan01/Pebbles/blob/master/figure/Accuracy_plot/pebble_unet_Split1D_image_accuracy.png?raw=true><img src=https://github.com/VitoChan01/Pebbles/blob/master/figure/Accuracy_plot/pebble_unet_Split1D_image_accuracy.png?raw=true width="100%" height="100%"></a>
        <figcaption>b: Split1D</figcaption>
    </figure>
</div>
<div style="display: flex; flex-direction: row; justify-content: center;">
    <figure style="flex: 1; margin-right: 10px;">
        <a href=https://github.com/VitoChan01/Pebbles/blob/master/figure/Accuracy_plot/pebble_unet_Split2C_image_accuracy.png?raw=true><img src=https://github.com/VitoChan01/Pebbles/blob/master/figure/Accuracy_plot/pebble_unet_Split2C_image_accuracy.png?raw=true width="100%" height="100%"></a>
        <figcaption>c: Split2C</figcaption>
    </figure>
    <figure style="flex: 1; margin-right: 0px;">
        <a href=https://github.com/VitoChan01/Pebbles/blob/master/figure/Accuracy_plot/pebble_unet_Split2D_image_accuracy.png?raw=true><img src=https://github.com/VitoChan01/Pebbles/blob/master/figure/Accuracy_plot/pebble_unet_Split2D_image_accuracy.png?raw=true width="100%" height="100%"></a>
        <figcaption>d: Split2D</figcaption>
    </figure>
</div>
<br/>
<center>
<figcaption>Figure 7: Model accuracy by image.</figcaption>
</center>

<div style="display: flex; flex-direction: row; justify-content: center;">
    <figure style="flex: 1; margin-right: 0px;">
        <a href=https://github.com/VitoChan01/Pebbles/blob/master/figure/Accuracy_plot/pebble_unet_Split1C_kfold_consistency.png?raw=true><img src=https://github.com/VitoChan01/Pebbles/blob/master/figure/Accuracy_plot/pebble_unet_Split1C_kfold_consistency.png?raw=true width="100%" height="100%"></a>
        <figcaption>a: Split1C</figcaption>
    </figure>
    <figure style="flex: 1; margin-right: 0px;">
        <a href=https://github.com/VitoChan01/Pebbles/blob/master/figure/Accuracy_plot/pebble_unet_Split1D_kfold_consistency.png?raw=true><img src=https://github.com/VitoChan01/Pebbles/blob/master/figure/Accuracy_plot/pebble_unet_Split1D_kfold_consistency.png?raw=true width="100%" height="100%"></a>
        <figcaption>b: Split1D</figcaption>
    </figure>
</div>
<div style="display: flex; flex-direction: row; justify-content: center;">
    <figure style="flex: 1; margin-right: 10px;">
        <a href=https://github.com/VitoChan01/Pebbles/blob/master/figure/Accuracy_plot/pebble_unet_Split2C_kfold_consistency.png?raw=true><img src=https://github.com/VitoChan01/Pebbles/blob/master/figure/Accuracy_plot/pebble_unet_Split2C_kfold_consistency.png?raw=true width="100%" height="100%"></a>
        <figcaption>c: Split2C</figcaption>
    </figure>
    <figure style="flex: 1; margin-right: 0px;">
        <a href=https://github.com/VitoChan01/Pebbles/blob/master/figure/Accuracy_plot/pebble_unet_Split2D_kfold_consistency.png?raw=true><img src=https://github.com/VitoChan01/Pebbles/blob/master/figure/Accuracy_plot/pebble_unet_Split2D_kfold_consistency.png?raw=true width="100%" height="100%"></a>
        <figcaption>d: Split2D</figcaption>
    </figure>
</div>
<br/>
<center>
<figcaption>Figure 8: Consistency over folds.</figcaption>
</center>

Through the union of SAM mask and U-Net mask, point cloud of individual pebbles can be created easily. Figure 9 presents an example created through open3d `geometry.create_point_cloud_from_depth_image`. Unfortunately, the example presented here was not properly scaled. For the pint cloud to be correctly scaled, 4 calibration parameters are necessary. These are the x,y focal length and principal points of the ZED 2i camera. These parameters are accessible through the camera API but unfortunately they were not exported and saved before returning the camera to the University. Nevertheless, it is possible to create point clouds for further analysis based on segmentation results. 

<center>
<figure style="flex: 1; margin-right: 0px;">
    <a href=https://github.com/VitoChan01/Pebbles/blob/master/figure/pebble_point_cloud.png?raw=true><img src=https://github.com/VitoChan01/Pebbles/blob/master/figure/pebble_point_cloud.png?raw=true width="80%" height="80%"></a>
    <figcaption>Figure 9: Point cloud of a pebble created from segmented depth map and RGB image </figcaption>
</figure>
</center>

# Conclusion and discussion


can do?
change camera parameter?
check accuracy of combined approach. give some measurement of extracted pebble size? or extract a pebble point cloud?
# Reference
Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., Dollár, P., & Girshick, R. (2023). Segment Anything. 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 3992–4003. https://doi.org/10.1109/ICCV51070.2023.00371

Mustafah, Y. M., Noor, R., Hasbi, H., & Azma, A. W. (2012). Stereo vision images processing for real-time object distance and size measurements. 2012 International Conference on Computer and Communication Engineering (ICCCE), 659–663. https://doi.org/10.1109/ICCCE.2012.6271270

Soloy, A., Turki, I., Fournier, M., Costa, S., Peuziat, B., & Lecoq, N. (2020). A Deep Learning-Based Method for Quantifying and Mapping the Grain Size on Pebble Beaches. Remote Sensing, 12(21), Article 21. https://doi.org/10.3390/rs12213659

Wang, C., Lin, X., & Chen, C. (2019). Gravel Image Auto-Segmentation Based on an Improved Normalized Cuts Algorithm. Journal of Applied Mathematics and Physics, 7(3), Article 3. https://doi.org/10.4236/jamp.2019.73044
