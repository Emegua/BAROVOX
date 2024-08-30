#!/bin/bash
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
python -m pip install wandb matplotlib tqdm pathlib IPython
