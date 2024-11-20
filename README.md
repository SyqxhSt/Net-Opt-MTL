# Net-Opt-MTL
A Multi Task Learning Model Applied to Computer Vision
## Network Framework Diagram (VMSANEt)

<p align="center">
  <img src="./VMSANet.png" width="700"/>
</p>

## 1. Preparation
We provide pytorch compressed files for Ubuntu environment (recommended to use virtual environment)

Python 3.8 (Ubuntu 18.04)

Cuda  11.1

### requirement
matplotlib==3.6.2

numpy==1.26.4

pandas==2.2.2

pillow==10.4.0

scikit-learn==1.4.2

scipy==1.13.1

torch==2.4.0

torchvision==0.19.0

tqdm==4.66.4

## 2. Datasets

* **CITYSCAPES**: The preprocessed (normalized) Cityscapes dataset. Click [Cityscape](https://www.dropbox.com/scl/fi/wfmmk8tjn631723e0ycwm/Cityscapes.zip?rlkey=eyjw0vg9l48yvg77g0hm69y7z&st=y40g2ivx&dl=0) to download.
  ```
  <Cityscapes>/                     % Cityscapes dataset root (128 × 256 × 3)
      |
      ├── train/
      |     ├── image/              % Input image
      |     ├── label/              % Semantic segmentation labels (7 categories)
      |     └── depth/              % Depth estimation label
      |
      └── val/
            ├── image/              % Input image
            ├── label/              % Semantic segmentation labels (7 categories)
            └── depth/              % Depth estimation label
  ```

  * **Nyuv2**: Indoor scene dataset. Click [Nyuv2](https://www.dropbox.com/scl/fi/dgwxetgkfepnplsc3772n/Nyuv2.zip?rlkey=e3gi4m39efullrfhg73d7yp86&st=0ym91vu7&dl=0) to download.
  ```
  <Nyuv2>/                          % Nyuv2 dataset root (288 × 384 × 3)
      |
      ├── train/
      |     ├── image/              % Input image
      |     ├── label/              % Semantic segmentation labels (13 categories)
      |     └── depth/              % Depth estimation label
      |
      └── val/
            ├── image/              % Input image
            ├── label/              % Semantic segmentation labels (13 categories)
            └── depth/              % Depth estimation label
  ```

    * **KITTI**: Outdoor scene synthesis dataset. Click [KITTI](https://www.dropbox.com/scl/fi/0xtpkoj7ynh57lrcoims9/KITTI.zip?rlkey=mfozh7w8lub16pqlxcwvywylk&st=773jb32h&dl=0) to download. (This is the dataset we processed.)
  ```
  <Nyuv2>/                          % KITTI dataset root (128 × 256 × 3)
      |
      ├── train/
      |     ├── image/              % Input image
      |     ├── label/              % Semantic segmentation labels (14 categories)
      |     └── depth/              % Depth estimation label
      |
      └── val/
            ├── image/              % Input image
            ├── label/              % Semantic segmentation labels (14 categories)
            └── depth/              % Depth estimation label
  ```

## 3. Program running

- Single task learning baseline:
  ```
  python Single_task.py --task=semantic/depth/...
  ```
  
- Other multi-tasks learning:
  ```
  python model_name.py
  ```
We also provide additional model downloads, including Cross-Stitch, MTRAN, Multi-Task, and Single-Task models. If needed, feel free to click [Other models](https://www.dropbox.com/scl/fi/y5je8fgfh3eq2gblbav84/Other_models.zip?rlkey=mhxmfqda6vuza3j22rxv2ngls&st=lima94gl&dl=0) to download them.
