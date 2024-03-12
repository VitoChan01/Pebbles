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
  - Stereo disparity
  - segmentation
  - classification
  - U-NET
---

# Introduction
General idea:\
do a semantic segmentation of pebbles see wt works better
why? search later

Sediment charateristics and grain-size distribution carries important information on the drainage system, the ecosystem, and the weather condition (Soloy et al., 2020; Wang et al., 2019). Unfortunately, traditional methods to manually collect these information are costly, labor intensive, and time consuming. Over the years, various techniques have been developed seeking to reduce manual input through machine learning. This project is an attempt to utilize U-NET and the Meta Segment-Anything Model in conjunction with stereo depth estimation to automize image based sampling. 

## Approach
### U-NET and Segment-Anything
just brief intro of unet
### Stereo imagery
discuss the interest of if depth info bring smt in 
3 questions:
How well U-NET can perform with limited training data and resolution. How well does a pretrained model perform in segmentation. Does the inclusion of a depth layer improve the performance. 




# Method
## Data collection

## Model construction

# Results
training history
# Accuracy

# Conclusion

# Reference

Soloy, A., Turki, I., Fournier, M., Costa, S., Peuziat, B., & Lecoq, N. (2020). A Deep Learning-Based Method for Quantifying and Mapping the Grain Size on Pebble Beaches. Remote Sensing, 12(21), Article 21. https://doi.org/10.3390/rs12213659\
Wang, C., Lin, X., & Chen, C. (2019). Gravel Image Auto-Segmentation Based on an Improved Normalized Cuts Algorithm. Journal of Applied Mathematics and Physics, 7(3), Article 3. https://doi.org/10.4236/jamp.2019.73044\
