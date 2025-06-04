<div><h2>[CVPR'25] SplineGS: Robust Motion-Adaptive Spline for Real-Time Dynamic 3D Gaussians from Monocular Video</h2></div>
<br>

**[Jongmin Park](https://sites.google.com/view/jongmin-park)<sup>1\*</sup>, [Minh-Quan Viet Bui](https://quan5609.github.io/)<sup>1\*</sup>, [Juan Luis Gonzalez Bello](https://sites.google.com/view/juan-luis-gb/home)<sup>1</sup>, [Jaeho Moon](https://sites.google.com/view/jaehomoon)<sup>1</sup>, [Jihyong Oh](https://cmlab.cau.ac.kr/)<sup>2†</sup>, [Munchurl Kim](https://www.viclab.kaist.ac.kr/)<sup>1†</sup>** 
<br>
<sup>1</sup>KAIST, South Korea, <sup>2</sup>Chung-Ang University, South Korea
<br>
\*Co-first authors (equal contribution), †Co-corresponding authors
<p align="center">
        <a href="https://kaist-viclab.github.io/splinegs-site/" target='_blank'>
        <img src="https://img.shields.io/badge/🐳-Project%20Page-blue">
        </a>
        <a href="https://openaccess.thecvf.com/content/CVPR2025/papers/Park_SplineGS_Robust_Motion-Adaptive_Spline_for_Real-Time_Dynamic_3D_Gaussians_from_CVPR_2025_paper.pdf" target='_blank'>
        <img src="https://img.shields.io/badge/2025-CVPR-brightgreen">
        </a>
        <a href="https://arxiv.org/abs/2412.09982" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2312.13528-b31b1b.svg">
        </a>
        <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/KAIST-VICLab/SplineGS">
</p>

<p align="center" width="100%">
    <img src="https://github.com/KAIST-VICLab/SplineGS/blob/main/assets/architecture.png?raw=tru"> 
</p>

## 📣 News
### Updates
- **May 26, 2025**: Code released.
- **February 26, 2025**: SplineGS accepted to CVPR 2025 🎉.
- **December 13, 2024**: Paper uploaded to arXiv. Check out the manuscript [here](https://arxiv.org/abs/2412.09982).(https://arxiv.org/abs/2412.09982).
### To-Dos
- Add DAVIS dataset configurations.
- Add custom dataset support.
- Add iPhone dataset configurations.
## Environmental Setups
Clone the repo and install dependencies:
```sh
git clone https://github.com/KAIST-VICLab/SplineGS.git --recursive
cd SplineGS

# install splinegs environment
conda create -n splinegs python=3.7 
conda activate splinegs
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install nvidia/label/cuda-11.7.0::cuda
conda install nvidia/label/cuda-11.7.0::cuda-nvcc
conda install nvidia/label/cuda-11.7.0::cuda-runtime
conda install nvidia/label/cuda-11.7.0::cuda-cudart


pip install -e submodules/simple-knn
pip install -e submodules/co-tracker
pip install -r requirements.txt

# install depth environment
conda deactivate
conda create -n unidepth_splinegs python=3.10
conda activate unidepth_splinegs

pip install -r requirements_unidepth.txt
conda install -c conda-forge ld_impl_linux-64
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
conda install nvidia/label/cuda-12.1.0::cuda
conda install nvidia/label/cuda-12.1.0::cuda-nvcc
conda install nvidia/label/cuda-12.1.0::cuda-runtime
conda install nvidia/label/cuda-12.1.0::cuda-cudart
conda install nvidia/label/cuda-12.1.0::libcusparse
conda install nvidia/label/cuda-12.1.0::libcublas
cd submodules/UniDepth/unidepth/ops/knn;bash compile.sh;cd ../../../../../
cd submodules/UniDepth/unidepth/ops/extract_patches;bash compile.sh;cd ../../../../../

pip install -e submodules/UniDepth
mkdir -p submodules/mega-sam/Depth-Anything/checkpoints
```
## 📁 Data Preparations
### Nvidia Dataset
1. We follow the evaluation setup from [RoDynRF](https://robust-dynrf.github.io/). Download the training images [here](https://github.com/KAIST-VICLab/SplineGS/releases/tag/dataset) and arrange them as follows:
```bash
SplineGS/data/nvidia_rodynrf
    ├── Balloon1
    │   ├── images_2
    │   ├── instance_masks
    │   ├── motion_masks
    │   └── gt
    ├── ...
    └── Umbrella
```
2. Download [Depth-Anything checkpoint](https://huggingface.co/spaces/LiheYoung/Depth-Anything/blob/main/checkpoints/depth_anything_vitl14.pth) and place it at `submodules/mega-sam/Depth-Anything/checkpoints`. Generate depth estimation and tracking results for all scenes as:
```sh
conda activate unidepth_splinegs
bash gen_depth.sh

conda deactivate
conda activate splinegs
bash gen_tracks.sh
```
3. To obtain motion masks, please refer to [Shape of Motion](https://github.com/vye16/shape-of-motion/). For Nvidia dataset, we provide the precomputed in `motion_masks` folder
### YOUR OWN Dataset
T.B.D
## 🚀 Get Started
### Nvidia Dataset
#### Training
```sh
# check if environment is activated properly
conda activate splinegs

python train.py -s data/nvidia_rodynrf/${SCENE}/ --expname "${EXP_NAME}" --configs arguments/nvidia_rodynrf/${SCENE}.py
```
#### Metrics Evaluation
```sh
python eval_nvidia.py -s data/nvidia_rodynrf/${SCENE}/ --expname "${EXP_NAME}" --configs arguments/nvidia_rodynrf/${SCENE}.py --checkpoint output/${EXP_NAME}/point_cloud/fine_best
```
### YOUR OWN Dataset
#### Training
T.B.D
#### Evaluation
T.B.D

## Acknowledgments
- This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korean Government [Ministry of Science and ICT (Information and Communications Technology)] (Project Number: RS-2022-00144444, Project Title: Deep Learning Based Visual Representational Learning and Rendering of Static and Dynamic Scenes, 100%).

## ⭐ Citing SplineGS

If you find our repository useful, please consider giving it a star ⭐ and citing our research papers in your work:
```bibtex
@InProceedings{Park_2025_CVPR,
    author    = {Park, Jongmin and Bui, Minh-Quan Viet and Bello, Juan Luis Gonzalez and Moon, Jaeho and Oh, Jihyong and Kim, Munchurl},
    title     = {SplineGS: Robust Motion-Adaptive Spline for Real-Time Dynamic 3D Gaussians from Monocular Video},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {26866-26875}
}
```
