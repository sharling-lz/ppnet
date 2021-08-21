# Estimating Human Pose Efficiently by Parallel Pyramid Networks
An efficient network architecture for human pose estimation

## News
[2021/7/1] Code and pretrained models are public available.

[2021/7/26] Our paper "Estimating Human Pose Efficiently by Parallel Pyramid Networks" is published on TIP.

## Introduction
Parallel Pyramid Network (PPNet) features separating spatial location preserving from semantic information acquisition in architecture design. With this unique design, superior performance is able to be delivered by using less parameters and GFLOPs comparing to the existing network architectures in the literature. </br>

![Illustrating the architecture of the proposed PPNet](/figures/PPNet-Arc.jpg)
## Main Results
### Results on MPII val
| Arch               |#Params | GFLOPs | Head | Shoulder | Elbow | Wrist |  Hip | Knee | Ankle | Mean | Mean@0.1 |
|--------------------|--------|--------|------|----------|-------|-------|------|------|-------|------|----------|
| SB ResNet-152      | 68.6M  | 21.0   | 97.0 |     95.9 |  90.0 |  85.0 | 89.2 | 85.3 |  81.3 | 89.6 |     35.0 |
| HRNet W32          | 28.5M  |  9.5   | 97.1 |     95.9 |  90.3 |  86.4 | 89.1 | 87.1 |  83.3 | 90.3 |     37.7 |
| **PPNet M4-D2-W32**| 12.7M  |  6.7   | 97.1 |     96.2 |  90.9 |  86.6 | 89.6 | 87.0 |  83.9 | 90.6 |     38.2 |

### Note:
- Flip test is used.
- Input size is 256x256
- All models are pretrained on ImageNet

### Results on COCO val2017
| Backbone           | Pretrain   |Input size | #Params | GFLOPs | Train speed  |    AP | Ap .5 | AP .75 | 
|--------------------|------------|-----------|---------|--------|------------- |-------|-------|--------|
| SB ResNet-50       |     Y      |   256x192 | 33.9M   |    8.9 |190 samples/s | 0.704 | 0.886 |  0.783 |
| SB ResNet-50       |     Y      |   384x288 |         |   20.2 | 90 samples/s | 0.722 | 0.893 |  0.789 |
| HRNet W32          |     Y      |   256x192 | 28.5M   |    7.1 |100 samples/s | 0.744 | 0.905 |  0.819 | 
| HRNet W32          |     Y      |   384x288 |         |   16.0 | 50 samples/s | 0.758 | 0.906 |  0.827 |
| HRNet W48          |     N      |   384x288 | 63.6M   |   32.9 | 32 samples/s | 0.750 | 0.900 |  0.819 | 
| **PPNet M2-D2-W32**|     Y      |   256x192 |  7.0M   |    3.3 |235 samples/s | 0.721 | 0.890 |  0.795 |
| **PPNet M2-D2-W32**|     Y      |   384x288 |         |    7.5 |110 samples/s | 0.738 | 0.897 |  0.804 | 
| **PPNet M4-D2-W32**|     Y      |   256x192 | 12.7M   |    5.0 |150 samples/s | 0.738 | 0.898 |  0.811 | 
| **PPNet M4-D2-W32**|     Y      |   384x288 |         |   11.3 | 75 samples/s | 0.757 | 0.901 |  0.822 |
| **PPNet M4-D2-W48**|     N      |   384x288 | 27.1M   |   20.6 | 53 samples/s | 0.755 | 0.898 |  0.821 |
| **PPNet M4-D3-W48**|     N      |   384x288 | 39.2M   |   28.1 | 42 samples/s | 0.758 | 0.900 |  0.824 |

### Note:
- Flip test is used.
- Person detector has person AP of 56.4 on COCO val2017 dataset.
- GFLOPs is for convolution and linear layers only.
- The training speed is obtained on a machine with one NVIDIA 2080Ti GPU

## Environment
The code is developed using python 3.6 on Ubuntu 18.04. NVIDIA GPUs are needed. The code is developed and tested using 1 NVIDIA 2080Ti GPU.

## Quick start
### Installation
1. Install pytorch >= v1.0.0 following [official instruction](https://pytorch.org/).
2. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Make libs:
   ```
   cd ${POSE_ROOT}/lib
   make
   ```
5. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.
4. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output 
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── tools 
   ├── README.md
   └── requirements.txt
   ```

6. Download pretrained models from ([GoogleDrive](https://drive.google.com/drive/folders/1_sNpgMHwMWn_19qjv_OKpzY-8eEyKZES?usp=sharing))
   ```
   ${POSE_ROOT}
    `-- models
        `-- pytorch
            |-- imagenet
            |   |-- ppnet_m2d2w32.pth.tar
            |   |-- ppnet_m4d2w32.pth.tar
            |-- pose_coco
            |   |-- pose_ppnet_M2-D2-W32_256x192.pth
            |   |-- pose_ppnet_M2-D2-W32_384x288.pth
            |   |-- pose_ppnet_M4-D2-W32_256x192.pth
            |   |-- pose_ppnet_M4-D2-W32_384x288.pth
            |   |-- pose_ppnet_M4-D2-W48_384x288.pth
            |   |-- pose_ppnet_M4-D3-W48_384x288.pth
            `-- pose_mpii
                |-- pose_ppnet_M4-D2-W32_256x256.pth

   ```
   
### Data preparation
**For MPII data**, please download from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/). The original annotation files are in matlab format. The json format, you need to download them from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW00SqrairNetmeVu4) or [GoogleDrive](https://drive.google.com/drive/folders/1En_VqmStnsXMdldXA6qpqEyDQulnmS3a?usp=sharing).
Extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- mpii
    `-- |-- annot
        |   |-- gt_valid.mat
        |   |-- test.json
        |   |-- train.json
        |   |-- trainval.json
        |   `-- valid.json
        `-- images
            |-- 000001163.jpg
            |-- 000003072.jpg
```

**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. You also need the person detection result of COCO val2017 and test-dev2017 to reproduce multi-person pose estimation results. Please download from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-) or [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing).
Download and extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
        |   |-- COCO_test-dev2017_detections_AP_H_609_person.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```

### Testing 
MPII for example
```
python tools/test.py --cfg experiments/mpii/ppnet/M4_D2_W32_256x256_rms_lr25e-4.yaml TEST.MODEL_FILE './models/pytorch/pose_mpii/pose_ppnet_M4-D2-W32_256x256.pth'
```

### Training

```
python tools/train.py --cfg experiments/mpii/ppnet/M4_D2_W32_256x256_rms_lr25e-4.yaml
```

### Citation
If you use our code or models in your research, please cite with:
```
@ARTICLE{zhao-ppnet,
  author={Zhao, Lin and Wang, Nannan and Gong, Chen and Yang, Jian and Gao, Xinbo},
  journal={IEEE Transactions on Image Processing}, 
  title={Estimating Human Pose Efficiently by Parallel Pyramid Networks}, 
  year={2021},
  volume={30},
  number={},
  pages={6785-6800},
  doi={10.1109/TIP.2021.3097836}}
```

### Acknowledgement
Our codes are developed based on the opensource of HRNet

[Deep High-Resolution Representation Learning for Human Pose Estimation, Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong](https://github.com/HRNet/HRNet-Human-Pose-Estimation)
