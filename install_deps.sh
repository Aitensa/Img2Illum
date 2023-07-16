#!/usr/bin/env sh




conda create -n Ex python=3.8 --yes
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Ex


conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install pytorch-lightning==1.2.8

conda install -c plotly psutil requests opencv-python dill python-kaleido --yes
pip install cython==0.29.20 autowrap ninja tables ply ilock
pip install h5py pydocstyle plotly psutil xvfbwrapper yapf mypy openmesh plyfile neuralnet-pytorch imageio pyinstrument pairing robust_laplacian pymesh trimesh cmake "ray[tune]" "pytorch-lightning-bolts>=0.2.5" pyrr gdist neptune-client neptune-contrib iopath sklearn autowrap py-goicp opencv-python torchsummary gdown
conda install "notebook>=5.3" "ipywidgets>=7.2" flake8 black flake8 -y
conda install pytorch-metric-learning -c metric-learning -c pytorch -y
pip install addict
pip install open3d-python
pip install git+git://github.com/fwilliams/point-cloud-utils
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
conda install open3d tifffile h5py fire imageio scipy pyrsistent jupyter -c conda-forge



pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse==0.6.7
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric


mkdir -p etc/torch
mkdir -p etc/torch_cluster

wget --directory-prefix etc/torch/ https://download.pytorch.org/whl/cu110/torchvision-0.8.2%2Bcu110-cp38-cp38-linux_x86_64.whl
wget --directory-prefix etc/torch/ https://download.pytorch.org/whl/cu110/torch-1.7.1%2Bcu110-cp38-cp38-linux_x86_64.whl
wget --directory-prefix etc/torch_cluster/ https://pytorch-geometric.com/whl/torch-1.7.0+cu110/torch_cluster-1.5.8-cp38-cp38-linux_x86_64.whl

# pipenv install
pip install etc/torch/torch-1.7.1+cu110-cp38-cp38-linux_x86_64.whl
pip install etc/torch/torchvision-0.8.2+cu110-cp38-cp38-linux_x86_64.whl
pip install etc/torch_cluster/torch_cluster-1.5.8-cp38-cp38-linux_x86_64.whl
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
