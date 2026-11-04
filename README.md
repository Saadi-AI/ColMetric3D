<p align="center">
  <h2 align="center">ColMetric3D: Scale-Consistent Monocular Colonoscopy 3D Reconstruction via Metric-Scale Expert Calibration</h2>
  <h3 align="center">
  </h3>

  <p align="center">
    <strong>Saad Khalil</strong><sup>1</sup>&nbsp;&nbsp;
    <strong>Han Hee Lee</strong><sup>2</sup>&nbsp;&nbsp;
    <strong>Youngbae Hwang</strong><sup>1</sup>
    <br>
    <sup>1</sup>Department of Intelligent Systems and Robotics, Chungbuk National University, South Korea
    <br>
    <sup>2</sup>Department of Internal Medicine, The Catholic University of Korea, South Korea
  </p>
</p>

---

This repository contains the official evaluation and inference code for **ColMetric3D**, a scale-consistent framework for native metric depth estimation, relative pose recovery, and 3D reconstruction in monocular colonoscopy.

ColMetric3D adapts a **Depth Anything V2** foundation backbone using **latent-aware singular-vector fine-tuning (CrG-LoRA)**, refines local colonoscopic depth structure with **Local Depth Distribution Refinement (LDDR)**, and performs spatially varying native metric calibration using **Metric-Scale Expert Calibration (MSEC)**. The framework predicts native metric depth under known camera intrinsics and uses metric-depth-conditioned pose estimation for monocular colonoscopy 3D reconstruction.

> **Note:** This release provides evaluation and inference code with pretrained weights. Training code is not included in the initial release.

---

## TODO

- [ ] Release evaluation code
- [ ] Release inference code
- [ ] Release training code
- [ ] Release project page

---

## Overview

ColMetric3D consists of two main components:

1. **DepthNet**: predicts native metric depth from a monocular colonoscopy frame using DAv2 + LA-SVFT, LDDR, and MSEC.
2. **PoseNet**: estimates relative camera motion from RGB frame pairs conditioned on predicted metric depth, camera intrinsics, and a reliability mask.

The predicted metric depths and relative poses are integrated by back-projection and pose accumulation to obtain sequence-level 3D reconstruction.

<p align="center">
  <img src="assets/overview.png" width="95%">
</p>

---

## Repository Structure
We are cleaning the code for release.
## Acknowledgements

This project builds upon recent advances in monocular depth estimation, foundation-model adaptation, and endoscopic 3D reconstruction. We thank the authors of C3VD, SimCol3D, Depth Anything V2, Metric3D, EndoDAC, Endo-FASt3R, and Endo3R for making their datasets, models, or code available to the research community.

---
