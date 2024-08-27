## Pay attention to cycle for spatio-temporal graph neural network

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

