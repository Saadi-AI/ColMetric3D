<p align="center">
  <h2 align="center">ColMetric3D: Scale-Consistent Monocular Colonoscopy 3D Reconstruction via Metric-Scale Expert Calibration</h2>
  <h3 align="center">IEEE Transactions on Medical Imaging</h3>
  <h3 align="center">
    <a href="PAPER_LINK">Paper</a> |
    <a href="PROJECT_PAGE_LINK">Project Page</a> |
    <a href="CHECKPOINT_LINK">Models</a>
  </h3>
  <p align="center">
    <strong>Saad Khalil</strong><sup>1</sup>&nbsp;&nbsp;
    <strong>Han Hee Lee</strong><sup>2</sup>&nbsp;&nbsp;
    <strong>Youngbae Hwang</strong><sup>1</sup>
    <br>
    <sup>1</sup>Chungbuk National University&nbsp;&nbsp;
    <sup>2</sup>The Catholic University of Korea
  </p>
</p>

This repository contains the evaluation and inference code for **ColMetric3D**, a scale-consistent framework for native metric depth estimation, relative pose recovery, and 3D reconstruction in monocular colonoscopy.

## TODO
- [x] Release evaluation code
- [x] Release inference code
- [x] Release pretrained weights
- [ ] Release training code
- [ ] Release project page

## Overview

ColMetric3D adapts a Depth Anything V2 backbone using LA-SVFT, refines local colonoscopic depth structure with LDDR, and performs spatially varying metric calibration using MSEC.

![overview](assets/overview.png)

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/ColMetric3D.git
cd ColMetric3D

conda create -n colmetric3d python=3.10
conda activate colmetric3d

pip install -r requirements.txt
```

## Download Checkpoints
```bash
mkdir -p checkpoints

# replace with your real link
wget -O checkpoints/colmetric3d_c3vd_depth.pth CHECKPOINT_DEPTH_URL
wget -O checkpoints/colmetric3d_c3vd_pose.pth CHECKPOINT_POSE_URL
```
## Prepare dATASTES


