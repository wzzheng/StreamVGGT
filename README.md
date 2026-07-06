<div align="center">
<h1>MeTRIC: Metric Temporally-consistent Reconstruction In Causal Streaming</h1>
</div>

### [Paper]TODO  | [Project Page]TODO | [Online Demo]TODO

>Metric Temporally-consistent Reconstrution In Causal Streaming

>Porter Dosch<sup>\*</sup>, Luise Haller<sup>\*</sup>, Faisal Zaghloul, James Tompkin

<sup>*</sup> Equal contribution.

**MeTRIC**, a causal transformer architecture for **temporally consistent 4D geometry generation** built off of [StreamVGGT](https://github.com/wzzheng/StreamVGGT), delivers both fast inference and temporally-consistent 4D reconstruction.

## News
TODO: as project progresses, update this section

## Overview
TODO: add overview of method once everything finalized

### Installation

1. Clone MeTRIC
```bash
git clone https://github.com/jPorterDosch/MeTRIC.git
cd MeTRIC
```
2. Create conda environment
```bash
conda create -n MeTRIC python=3.11 cmake=3.14.0
conda activate MeTRIC 
```

3. Install requirements
```bash
pip install -r requirements.txt
conda install 'llvm-openmp<16'
```

### Download Checkpoints
Pre-trained StreamVGGT is also available at both [Hugging Face](https://huggingface.co/lch01/StreamVGGT/) and [Tsinghua cloud](https://cloud.tsinghua.edu.cn/d/d6ad8f36fcd541bcb246/).

To download from huggingface, after installing `requirements.txt`, run
```
hf download lch01/StreamVGGT \
  --local-dir ./StreamVGGT
```

### Logging (Weights & Biases)

Training runs are logged to [wandb](https://wandb.ai). Set your API key
(from https://wandb.ai/authorize) before launching:

    export WANDB_API_KEY=your_key_here

To run without logging, set `WANDB_MODE=offline` or `WANDB_MODE=disabled`.

## Data Preparation
### Training Datasets
#### ARKitScenes
Download the raw data using the script provided by Apple
`bash download_arkit_scenes.sh`

Download the precomputed pairs provided by DUST3R
```
mkdir -p ~/scratch/data/arkit_scenes
cd ~/scratch/data/arkit_scenes

wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/arkitscenes_pairs.zip

unzip arkitscenes_pairs.zip
```

Then run:
```
python preprocess_arkitscenes.py --arkitscenes_dir /path/to/your/raw/data --precomputed_pairs /path/to/your/pairs --output_dir /path/to/your/outdir

python generate_set_arkitscenes.py --root /path/to/your/outdir --splits Training Test --max_interval 5.0 --num_workers 8
```

### Evaluation Datasets
Please refer to [MonST3R](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md) and [Spann3R](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md) to prepare Sintel, Bonn, KITTI, NYU-v2, ScanNet, 7scenes and Neural-RGBD datasets. 

For Sintel, Bonn, and KITTI, download scripts are available in `datasets_download`; preprocessing scripts are available in `datasets_preprocess`. These scripts are taken directly from the MONST3R repo for ease of use.
Download: `bash datasets_download/download_<name>.sh`
Preprocess: `python datasets_preprocess prepare_<name>.sh`
Sintel preprocessing is omitted since it is not necessary.

## Folder Structure
The overall folder structure should be organized as follows：
TODO: fill in rest of this once we have decided on directory structure
```
MeTRIC
├── ckpt/
|   ├── model.pt
|   └── checkpoints.pth
|
└── src/
    ├── ...
```

## Evaluation
TODO: add eval scripts

## Acknowledgements
Our code is based on the following repositories:

[StreamVGGT](https://github.com/wzzheng/streamvggt)
[DUSt3R](https://github.com/naver/dust3r)
[MonST3R](https://github.com/Junyi42/monst3r.git)
[Spann3R](https://github.com/HengyiWang/spann3r.git)
[CUT3R](https://github.com/CUT3R/CUT3R)
[VGGT](https://github.com/facebookresearch/vggt)
[Point3R](https://github.com/YkiWu/Point3R)

## Citation
TODO: add citation after any paper is written/uploaded
