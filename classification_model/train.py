import torch
import torch.nn as nn
import seaborn as sn
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys, pdb
import wandb
import os

from argparse import ArgumentParser
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

import core.dataloader
from core.dataloader import audioMNIST
from core.models import M5, M11
from core.dataloader import collate_fn_cnn, collate_fn_resnet
import core.models
from core.Trainer import ResnetTrainer, CNNTrainer

'''
Example:
     python train.py --train --model="resnet" --data_type "speechcommand" --wandb_enable 
            --n_fft=256 --sample_rate=2000 --model_path="saved_model/spect/test_model" 
            --epoch 20 --lr 0.0001 --wandb_entity cca-pressure --wandb_project finetuning
    python train.py --train --model="cnn" --data_type "speechcommand" --wandb_enable 
            --sample_rate=2000 --model_path="saved_model/spect/test_model_cnn"
            --epoch 20 --lr 0.0001  --wandb_entity cca-pressure --wandb_project finetuning
'''

class Config:
    '''Configuration and Argument Parser for script to train the classification.'''
    def __init__(self, args):
        self.parser = ArgumentParser(description='The parameters for training the scene graph using GCN.')
        self.parser.add_argument('--model', type=str, default='cnn', help="Select Model.")
        self.parser.add_argument('--data_path', type=str, default='/media/data1/yonatan/speechcommandDataset/pressureWavModified', help="Path to dataset.")
        self.parser.add_argument('--data_type', type=str, default='mnist', help="mnist/speechcommand")
        self.parser.add_argument("--scale", type=int, default=1, help="model width will be multiplied by scale") #diff
        self.parser.add_argument('--batch_size', type=int, default=256, help="batch size")
        self.parser.add_argument('--seed', type=int, default=1, help="batch size")
        self.parser.add_argument("--epoch", type=int, default=1, help="number of epochs to train")
        self.parser.add_argument("--log_interval", type=int, default=10, help="display train loss after every `log-interval` batch")
        self.parser.add_argument("--model_path", type=str, default="saved_model/spect/model_t", help="file to save model checkpoint")#rm
        self.parser.add_argument("--optimizer", type=str, default="adam", help="optimizer adam/sgd")
        self.parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
        self.parser.add_argument("--subspectral_norm", type=bool, default=False, help="use SubspectralNorm or Dropout")
        self.parser.add_argument('--train', action='store_true', help='train') 
        self.parser.add_argument('--test', action='store_true', help='test')
        #audio cfg
        self.parser.add_argument('--n_fft', type=int, default=256, help='test')
        self.parser.add_argument('--sample_rate', type=int, default=2200, help='test')
        self.parser.add_argument('--EPS', type=float, default=1e-9, help='test')
        self.parser.add_argument('--win_length', type=int, default=64, help='test')
        self.parser.add_argument('--hop_length', type=int, default=16, help='test')
        self.parser.add_argument("--model_file", type=str, help="path to model weights")
        self.parser.add_argument("--wav_file", type=str, help="path to model weights")
        
        #wandb settings
        self.parser.add_argument('--wandb_enable', action='store_true', help='Use wandb')
        self.parser.add_argument('--wandb_project', type=str, default="finetuning", help="wandb project name.")
        self.parser.add_argument('--wandb_entity', type=str, default="cca-pressure", help="wandb project entity.")

        #Hyperparameters
        self.parser.add_argument('--lr', type=float, default=0.005, help='test')
        self.parser.add_argument('--weight_decay', type=float, default=0.0001, help='test')
        
        args_parsed = self.parser.parse_args(args)
        
        for arg_name in vars(args_parsed):
            self.__dict__[arg_name] = getattr(args_parsed, arg_name)
        
        self.labels = [str(i) for i in range(10)] if self.data_type == 'mnist' else ["backward","cat","eight","forward","happy","left","no","one","seven","stop","tree","yes", "bed","dog","five","four","house","marvin","off","sheila","two","visual","zero", "bird","down","follow","go","learn","nine","on","right","six","three","up","wow"]

        if self.wandb_enable:
            self.wandb_ = wandb.init(project=self.wandb_project, entity=self.wandb_entity, name=f"{self.model}_batch-{self.batch_size}_seed-{self.seed}_epoch-{self.epoch}_nfft-{self.n_fft}")
            self.wandb_config = wandb.config
        else: 
            self.wandb_ = None
            self.wandb_config = None

        self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")



def dataloader(cfg):

    if cfg.device == "cuda":
        num_workers = 12
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    train_loader = torch.utils.data.DataLoader(
        audioMNIST(root=cfg.data_path, subset="training", data_type=cfg.data_type, model=cfg.model),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn_resnet if cfg.model=='resnet' else collate_fn_cnn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    validation_loader = torch.utils.data.DataLoader(
        audioMNIST(root=cfg.data_path, subset="validation", data_type=cfg.data_type, model=cfg.model),
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn_resnet if cfg.model=='resnet' else collate_fn_cnn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = torch.utils.data.DataLoader(
        audioMNIST(root=cfg.data_path, subset="testing", data_type=cfg.data_type, model=cfg.model),
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn_resnet if cfg.model=='resnet' else collate_fn_cnn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return train_loader, validation_loader, test_loader



def train_resnet(cfg):
    print(f"Device: {cfg.device}")
    print(f"Use subspectral norm: {cfg.subspectral_norm}")

    ml_model = core.models.BcResNetModel(
        n_class=len(cfg.labels),
        scale=cfg.scale,
        dropout=cfg.dropout,
        use_subspectral=cfg.subspectral_norm,
    ).to(cfg.device)
    
    trainer = ResnetTrainer(cfg, ml_model)
    trainer.train(ml_model, train_loader, validation_loader, test_loader)

def test_resnet(cfg):
    if not os.path.exists(cfg.model_file):
        raise FileExistsError(f"model {cfg.model_file} not exists")

    print(f"Device: {cfg.device}")
    print(f"Use subspectral norm: {cfg.subspectral_norm}")

    model = core.models.BcResNetModel(
        n_class=len(cfg.labels),
        scale=cfg.scale,
        dropout=cfg.dropout,
        use_subspectral=cfg.subspectral_norm,
    ).to(cfg.device)
    model.load_state_dict(torch.load(cfg.model_file))
    
    trainer = ResnetTrainer(cfg, model)
    trainer.test(model, test_loader)



def apply_command(cfg):
    
    if not os.path.exists(cfg.model_file):
        raise FileExistsError(f"model file {cfg.model_file} not exists")
    if not os.path.exists(cfg.wav_file):
        raise FileExistsError(f"sound file {cfg.wav_file} not exists")

    model = core.models.BcResNetModel(
        n_class=len(cfg.labels),
        scale=cfg.scale,
        dropout=cfg.dropout,
        use_subspectral=cfg.subspectral_norm,
    ).to(cfg.device)
    model.load_state_dict(torch.load(cfg.model_file))
    model.eval()
    trainer = ResnetTrainer(cfg, model)
    predictions = trainer.apply_to_file(model, cfg.wav_file, cfg.device)
    for label, prob in predictions[:5]:
        print(f"{label}\t{prob:.5f}")


def train_cnn(cfg):
    model = M11(n_input=1, n_output=len(cfg.labels))
    model.to(cfg.device)
    print(model)
    trainer = CNNTrainer(cfg, model, cfg.labels,cfg.device)
    trainer.train(train_loader, validation_loader)
  

def test_cnn(cfg):
    model = M11(n_input=1, n_output=len(cfg.labels))
    model.to(cfg.device)
    print(model)
    model.load_state_dict(torch.load(cfg.model_file))
    trainer = CNNTrainer(cfg, model, cfg.labels, cfg.device)
    trainer.evaluate(test_loader, cfg.epoch)


if __name__ == "__main__":
    cfg = Config(sys.argv[1:])
    core.dataloader.cfg = cfg
    train_loader, validation_loader, test_loader = dataloader(cfg)
    
    
    if cfg.model == 'cnn': 
        
        if cfg.train:
            train_cnn(cfg)
        if cfg.test:
            test_cnn(cfg)
    elif cfg.model == 'resnet':
        if cfg.train:
            train_resnet(cfg)
        if cfg.test:
            test_resnet(cfg)
        
