# Img2illum: Equivariant Indoor Illumination Map Estimation from A Single Image 


This is the official code release for *Equivariant Indoor Illumination Map Estimation from A Single Image* which was published in ECCV 2020. 



## Overview 
Thanks to the recent development of inverse rendering, photorealistic re-synthesis of indoor scenes have brought augmented reality closer to reality. All-angle environment illumination map estimation of arbitrary locations, as a fundamental task in this domain, is still challenging to deploy due to the requirement of expensive depth input. As such, we revisit the appealing setting of illumination estimation from a single image, using a cascaded formulation. The first stage predicts faithful depth maps from a single RGB image using a distortion-aware architecture. The second stage applies point cloud convolution operators that are equivariant to SO(3) transformations. These two technical ingredients collaborate closely with each other, because equivariant convolution would be meaningless without distortion-aware depth estimation. Using the public Matterport3D dataset, we demonstrate the effectiveness of our illumination estimation method both quantitatively and qualitatively. 

## Paper 

[*Equivariant Indoor Illumination Map Estimation from A Single Image*]().

If you use the Img2illum data or code, please cite: 

```bibtex
@InProceedings{pointar_eccv2020,
    author="",
    title="Equivariant Indoor Illumination Map Estimation from A Single Image",
    booktitle="CICAI",
    year="2023",
}
```


## How to use the repo

First, clone the repo.

```bash
git clone git@github.com:Aitensa/Img2Illum.git
cd Img2Illum
```

Then, install all the dependencies with `pipenv`:

```bash
pipenv install
pipenv shell


./install_deps.sh
```

## Preprocess Steps



To reproduce the work and generate the transformed point cloud datasets, please follow these steps:

1. Obtain access to the two open-source datasets: Matterport3D and Neural Illumination. You can find the Matterport3D dataset at [https://github.com/niessner/Matterport](https://github.com/niessner/Matterport), and the Neural Illumination dataset at [https://illumination.cs.princeton.edu](https://illumination.cs.princeton.edu).

2. Download the datasets to a directory of your choice. For the Matterport3D dataset, unzip the downloaded zip files and place them in a directory with a structure similar to `v1/scans/<SCENE_ID>/...`. For the Neural Illumination dataset, store the downloaded zip files (e.g., `illummaps_<SCENE_ID>.zip`) directly in a directory.

3. Open the `config.py` file located in the `datasets/pointar` directory. Modify the corresponding path variable in the file to reflect the local directory where you stored the datasets. Update the paths for both the Matterport3D and Neural Illumination datasets accordingly.

4. Once you have set the correct paths in the `config.py` file, you can proceed to generate the transformed point cloud datasets. Use the `gen_data.py` script provided in the `datasets/pointar` directory. Execute the script to start the generation process.

Please note that generating the entire dataset can take a significant amount of time, potentially a few hours, depending on the GPU devices available on your system. Additionally, keep in mind that the dataset size is approximately 1.4TB, so ensure you have enough storage space available.


## Model Training

To train the model:

```
python train.py

# For getting help info of the training script
python train.py --help
```
Our point cloud training component leverages the [PointConv](https://github.com/DylanWusee/pointconv_pytorch), the depth estimation leverages the [LeReS](https://github.com/aim-uofa/AdelaiDepth/tree/main/LeReS)




