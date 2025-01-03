## Enhancing Topological Dependencies in Spatio-Temporal Graphs with Cycle Message Passing Blocks

[![arXiv](https://img.shields.io/badge/arXiv-2402.02005-b31b1b.svg)](https://arxiv.org/abs/2401.15894)

This repository is the official implementation of [Enhancing Topological Dependencies in Spatio-Temporal Graphs with Cycle Message Passing Blocks](https://arxiv.org/abs/2402.02005) (Lee et al., LoG 2024).

## Environment Setup

```
conda create -n cy2mixer_env python=3.10
conda activate cy2mixer_env
conda install pytorch=1.13 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg=2.2 -c pyg -c conda-forge
pip install pyg-lib -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install matplotlib
pip install networkx
pip install pandas
pip install torchinfo
pip install pyyaml
```

## Training Commands Example

```bash
cd ./model
python train.py -d pems04 -g 0 -cfg Cy2Mixer
```

`<dataset>`:
- PEMS04
- PEMS07
- PEMS08

## Running Experiments on METR-LA

To conduct experiments on the METR-LA dataset, navigate to the Bigscity-LibCity directory and execute the following command:

```
cd ./Bigscity-LibCity
python run_model.py --task traffic_state_pred --model Cy2Mixer --dataset METR_LA
```

Ensure that the METR-LA data is located in the following directory:
```
./Bigscity-Libcity/raw_data/METR-LA
```

## About This Project
This project is developed and maintained by [SolverX](https://www.solverx.ai/) (previously known as Voltwin/AIAE).
