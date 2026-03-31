apt update && apt install -y wget bzip2

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh

source ~/.bashrc

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create -n splinegs python=3.10 -y
conda activate splinegs

pip install -r requirements.txt
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -U pip setuptools wheel ninja
apt update && apt install -y build-essential
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0
pip install --no-build-isolation ./submodules/simple-knn
apt update && apt install -y ffmpeg
# mkdir -p data/test_clip/images
# ffmpeg -i crowd9.mp4 -frames:v 100 -q:v 1 data/test_clip/images/%05d.png
# ls data2/test_clip/images/ | wc -l