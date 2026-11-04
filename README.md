<p align="center">
  <h2 align="center">ColMetric3D: Scale-Consistent Monocular Colonoscopy 3D Reconstruction via Metric-Scale Expert Calibration</h2>
  <h3 align="center">
    <a href="PAPER_LINK">Paper</a> |
    <a href="PROJECT_PAGE_LINK">Project Page</a> |
    <a href="CHECKPOINT_LINK">Pretrained Models</a>
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

ColMetric3D adapts a **Depth Anything V2** foundation backbone using **latent-aware singular-vector fine-tuning (LA-SVFT)**, refines local colonoscopic depth structure with **Local Depth Distribution Refinement (LDDR)**, and performs spatially varying native metric calibration using **Metric-Scale Expert Calibration (MSEC)**. The framework predicts native metric depth under known camera intrinsics and uses metric-depth-conditioned pose estimation for monocular colonoscopy 3D reconstruction.

> **Note:** This release provides evaluation and inference code with pretrained weights. Training code is not included in the initial release.

---

## TODO

- [x] Release evaluation code
- [x] Release inference code
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

```text
ColMetric3D/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── assets/
│   └── overview.png
├── checkpoints/
│   └── README.md
├── src/
│   ├── datasets.py
│   ├── datasets_simcol.py
│   ├── datasets_zeroshot_endoscopy.py
│   ├── checkpoint_utils.py
│   ├── metrics.py
│   ├── backend_opt.py
│   ├── eval_depth.py
│   ├── eval_pose_full.py
│   ├── eval_pose_simcol.py
│   ├── real_endo_infer_colmetric3d.py
│   ├── reconstruct_sequence.py
│   ├── viz_depth.py
│   └── networks/
│       ├── __init__.py
│       ├── backbone.py
│       ├── endometric3d_model.py
│       ├── moe_layer.py
│       ├── moe_router.py
│       └── pose_net.py
└── scripts/
    ├── eval_c3vd_depth.sh
    ├── eval_c3vd_pose.sh
    ├── eval_simcol_depth.sh
    └── infer_clinical.sh
````

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/ColMetric3D.git
cd ColMetric3D
```

Create a conda environment:

```bash
conda create -n colmetric3d python=3.10
conda activate colmetric3d
```

Install dependencies:

```bash
pip install -r requirements.txt
```

A typical `requirements.txt` is:

```text
torch>=2.1
torchvision
numpy
opencv-python
Pillow
tqdm
matplotlib
scipy
scikit-image
timm
einops
open3d
```

Install the CUDA-compatible PyTorch version for your system from the official PyTorch website if needed.

---

## Download Pretrained Checkpoints

Please download the pretrained ColMetric3D checkpoints from the following links:

| Model                          | Description                  | Link                      |
| ------------------------------ | ---------------------------- | ------------------------- |
| `colmetric3d_c3vd_depth.pth`   | C3VD depth checkpoint        | CHECKPOINT_DEPTH_LINK     |
| `colmetric3d_c3vd_pose.pth`    | C3VD pose checkpoint         | CHECKPOINT_POSE_LINK      |
| `colmetric3d_simcol_depth.pth` | SimCol depth checkpoint      | CHECKPOINT_SIMCOL_LINK    |
| `c3vd_msec_prototypes_k4.npz`  | Optional MSEC prototype file | CHECKPOINT_PROTOTYPE_LINK |

Place all checkpoint files under:

```text
checkpoints/
├── colmetric3d_c3vd_depth.pth
├── colmetric3d_c3vd_pose.pth
├── colmetric3d_simcol_depth.pth
└── c3vd_msec_prototypes_k4.npz
```

If the prototype file is not required by your checkpoint, it can be omitted.

---

## Dataset Preparation

### C3VD

Download C3VD from the official dataset page and organize it as:

```text
datasets/
└── c3vd_dataset/
    └── c3vd_test/
        ├── cecum_t1_a/
        ├── cecum_t2_a/
        ├── sigmoid_t3_a/
        ├── transverse_t4_a/
        └── ...
```

C3VD is used for native metric depth, pose, and 3D reconstruction evaluation.

### SimCol / SimCol3D

Organize SimCol as:

```text
datasets/
└── simcol/
    ├── train/
    ├── val/
    └── test/
```

SimCol is used for synthetic-domain native metric depth evaluation.

### Clinical Data

Private clinical data are **not included** in this repository.

For inference on your own colonoscopy video frames, organize frames as:

```text
examples/
└── clinical_frames/
    ├── 000000.png
    ├── 000001.png
    ├── 000002.png
    └── ...
```

Please ensure that all clinical data are de-identified and used under the appropriate institutional approval.

---

## Evaluation

### C3VD Depth Evaluation

```bash
python src/eval_depth.py \
  --dataset c3vd \
  --c3vd_test_root datasets/c3vd_dataset/c3vd_test \
  --ckpt_dir checkpoints/colmetric3d_c3vd_depth.pth \
  --height 476 \
  --width 588 \
  --batch_size 8
```

Expected output includes standard monocular depth metrics:

* AbsRel
* SqRel
* RMSE
* RMSElog
* δ1, δ2, δ3
* Mean Log-Scale error (MLS)

ColMetric3D is evaluated in native metric scale without post-hoc median scaling.

---

### C3VD Pose Evaluation

```bash
python src/eval_pose_full.py \
  --dataset c3vd \
  --c3vd_test_root datasets/c3vd_dataset/c3vd_test \
  --ckpt_dir checkpoints/colmetric3d_c3vd_pose.pth \
  --height 476 \
  --width 588
```

Expected output includes:

* APE
* RPE
* recovered Sim(3) scale factor

---

### SimCol Depth Evaluation

```bash
python src/eval_depth.py \
  --dataset simcol \
  --simcol_test_root datasets/simcol/test \
  --ckpt_dir checkpoints/colmetric3d_simcol_depth.pth \
  --height 476 \
  --width 476 \
  --batch_size 8
```

---

## Inference on Colonoscopy Frames

Run metric depth inference on a folder of colonoscopy frames:

```bash
python src/real_endo_infer_colmetric3d.py \
  --input_dir examples/clinical_frames \
  --output_dir outputs/clinical_demo \
  --ckpt_dir checkpoints/colmetric3d_c3vd_depth.pth \
  --height 476 \
  --width 588
```

The output folder will contain predicted depth maps and visualization panels.

---

## 3D Reconstruction

Run depth and pose inference followed by 3D reconstruction:

```bash
python src/reconstruct_sequence.py \
  --input_dir examples/clinical_frames \
  --output_dir outputs/reconstruction \
  --depth_ckpt checkpoints/colmetric3d_c3vd_depth.pth \
  --pose_ckpt checkpoints/colmetric3d_c3vd_pose.pth \
  --height 476 \
  --width 588
```

The reconstructed point cloud or mesh will be saved under:

```text
outputs/reconstruction/
```

Camera intrinsics should be known or estimated before reconstruction. If your sequence uses a different resolution or camera model, please resize intrinsics consistently with image preprocessing.

---

## Example Shell Scripts

You may also run the provided scripts:

```bash
bash scripts/eval_c3vd_depth.sh
bash scripts/eval_c3vd_pose.sh
bash scripts/eval_simcol_depth.sh
bash scripts/infer_clinical.sh
```

Before running, edit dataset and checkpoint paths inside each script.

---

## Notes and Limitations

* ColMetric3D assumes known or pre-calibrated camera intrinsics.
* The released checkpoints are intended for research use.
* The code does not include private clinical data.
* In-vivo colonoscopy examples should be interpreted qualitatively unless calibrated ground truth is available.
* Clinical frames must be de-identified before use.
* The current release focuses on evaluation and inference. Training code may be released later.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{khalil2026colmetric3d,
  title={ColMetric3D: Scale-Consistent Monocular Colonoscopy 3D Reconstruction via Metric-Scale Expert Calibration},
  author={Khalil, Saad and Lee, Han Hee and Hwang, Youngbae},
  journal={IEEE Transactions on Medical Imaging},
  year={2026}
}
```

---

## Acknowledgements

This project builds upon recent advances in monocular depth estimation, foundation-model adaptation, and endoscopic 3D reconstruction. We thank the authors of C3VD, SimCol3D, Depth Anything V2, Metric3D, EndoDAC, Endo-FASt3R, and Endo3R for making their datasets, models, or code available to the research community.

---

## License

This repository is released for research use. Please see `LICENSE` for details.

```
```
