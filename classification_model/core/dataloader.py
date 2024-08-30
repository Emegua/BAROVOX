import os
import math
import random
import torch
import torchaudio
from torchaudio import transforms
from typing import Optional, Tuple, Union
from torch.utils.data import Dataset
from pathlib import Path
from torch import Tensor



cfg = None
# default labels from GSC dataset





#Different
def prepare_wav(waveform, sample_rate):
    if sample_rate != cfg.sample_rate: 
        resampler = transforms.Resample(orig_freq=sample_rate, new_freq=cfg.sample_rate)
        waveform = resampler(waveform)
    to_mel = transforms.Spectrogram(n_fft=cfg.nfft, win_length=cfg.win_length, hop_length=cfg.hop_length)
    log_mel = (to_mel(waveform) + cfg.EPS).log2()
    return log_mel

def _load_list(root, *filenames):
    output = []
    for filename in filenames:
        filepath = os.path.join(root, filename)
        with open(filepath) as fileobj:
            output += [os.path.normpath(os.path.join(root, line.strip())) for line in fileobj]
    return output


#Done
def _load_waveform(
    root: str,
    filename: str,
    exp_sample_rate: int,
):
    path = os.path.join(root, filename)
    waveform, sample_rate = torchaudio.load(path)
    if exp_sample_rate != sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=exp_sample_rate)(waveform)
    return waveform


def _get_audioMNIST_metadata(filepath: str, path: str) -> Tuple[str, int, str, str, int]:
    #relpath = os.path.relpath(filepath, path)
    _, filename = os.path.split(filepath)         
    label, speaker_id, utterance_number = os.path.splitext(filename)[0].split("_")
    return filepath, cfg.sample_rate, label, speaker_id, utterance_number

def _get_speechCommand_metadata(filepath: str, path: str) -> Tuple[str, int, str, str, int]:
    #relpath = os.path.relpath(filepath, path)
    dirname, filename = os.path.split(filepath) 
    parentDirname, label_name = os.path.split(dirname)  
    speaker_id,_, utterance_number = os.path.splitext(filename)[0].split("_")   
    return filepath, cfg.sample_rate, label_name, speaker_id, utterance_number




class audioMNIST(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        subset: Optional[str] = None,
        data_type: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        
        if subset not in ["training", "validation", "testing"]:
            raise ValueError("Subset must be ['training', 'validation', 'testing'].")
        assert model in ['resnet', 'cnn']
        assert data_type in ['mnist', 'speechcommand']



        #seed 0 - - self.to_mel = transforms.Spectrogram(n_fft=N_FFT)
        self._path = os.fspath(root)

        if model == 'resnet':
            self.to_mel = transforms.Spectrogram(n_fft=cfg.n_fft, win_length=cfg.win_length, hop_length=cfg.hop_length)
            self._noise = []

        
        self.data_type = data_type
        self.model = model

        if not os.path.exists(self._path):
            raise RuntimeError(
                f"The path {self._path} doesn't exist. "
                "Please check the ``root`` path"
            )

        

        if subset == "validation":
            self._walker = _load_list(self._path, "validation_list.txt")
        elif subset == "testing":
            self._walker = _load_list("testing_list.txt")
        elif subset == "training":

            excludes = set(_load_list(self._path, "validation_list.txt", "testing_list.txt"))
            self._walker = sorted(str(p) for p in Path(self._path).glob("*/*.wav"))

            if self.model == 'cnn':
                self._walker = [w for w in self._walker if os.path.normpath(w) not in excludes]
            else:
                self._walker = [w for w in self._walker if w not in excludes]
        else:
            if self.model == 'resnet':
                raise ValueError(f"Unknown subset {subset}. Use validation/testing/training")
            else: 
                self._walker = sorted(str(p) for p in Path(self._path).glob("*/*.wav"))
        self.length = len(self._walker)

    
    




    #specific to resnet
    def _noise_augment(self, waveform):
        noise_waveform = random.choice(self._noise)

        noise_sample_start = 0
        if noise_waveform.shape[1] - waveform.shape[1] > 0:
            noise_sample_start = random.randint(0, noise_waveform.shape[1] - waveform.shape[1])
        noise_waveform = noise_waveform[:, noise_sample_start:noise_sample_start+waveform.shape[1]]

        signal_power = waveform.norm(p=2)
        noise_power = noise_waveform.norm(p=2)

        snr_dbs = [20, 10, 3]
        snr = random.choice(snr_dbs)

        snr = math.exp(snr / 10)
        scale = snr * noise_power / signal_power
        noisy_signal = (scale * waveform + noise_waveform) / 2
        return noisy_signal

    def _shift_augment(self, waveform):
        shift = random.randint(-1600, 1600)
        waveform = torch.roll(waveform, shift)
        if shift > 0:
            waveform[0][:shift] = 0
        elif shift < 0:
            waveform[0][shift:] = 0
        return waveform

    def _augment(self, waveform):
        if random.random() < 0.8:
            waveform = self._noise_augment(waveform)
        
        waveform = self._shift_augment(waveform)

        return waveform
    #specific to resnet
    
    
    #done
    def get_metadata(self, n: int) -> Tuple[str, int, str, str, int]:
        """Get metadata for the n-th sample from the dataset. Returns filepath instead of waveform,
        but otherwise returns the same fields as :py:func:`__getitem__`.

        Args:
            n (int): The index of the sample to be loaded
        Returns:
            Tuple of the following items;
            str:  Path to the audio
            int:  Sample rate
            str:  Label
            str:  Speaker ID
            int:  Utterance number
        """
        fileid = self._walker[n]
        if self.data_type == "mnist":
            return _get_audioMNIST_metadata(fileid, self._path)
        else: 
            return _get_speechCommand_metadata(fileid, self._path)

    #done
    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded
        Returns:
            Tuple of the following items;
            Tensor: Waveform
            int: Sample rate
            str: Label
            str: Speaker ID
            int: Utterance number
        """
        metadata = self.get_metadata(n)
        waveform = _load_waveform(self._path, metadata[0], metadata[1])
        
        if self.model == 'resnet':
            # if self._walker == "training":
        #        waveform = self._augment(waveform)
            log_mel = (self.to_mel(waveform) + cfg.EPS).log2()
            return log_mel, metadata[2]
        else:
            return (waveform,) + metadata[1:]
    
    
    def __len__(self) -> int:
        return self.length

#removed other index_to_label function


#combined the two label to index functions
def label_to_index(word):
    # Return the position of the word in labels
    _label_to_idx = {label: i for i, label in enumerate(cfg.labels)}
    if cfg.model == 'resnet':
        return _label_to_idx[word]
    return torch.tensor(_label_to_idx[word])


#replace in trainer
def pad_sequence_resnet(batch):
    new_batch = []
    for item in batch:
        try:
            new_batch.append(item.permute(2, 1, 0))
        except Exception as e:
            print(item.size())
            raise e

    batch = torch.nn.utils.rnn.pad_sequence(new_batch, batch_first=True)
    return batch.permute(0, 3, 2, 1) 


def pad_sequence_cnn(batch):
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    #pdb.set_trace()
    return batch.permute(0, 2, 1)


def collate_fn_resnet(batch):
    tensors, targets = [], []
    for log_mel, label in batch:
        tensors.append(log_mel)
        targets.append(label_to_index(label))
    tensors = pad_sequence_resnet(tensors)
    targets = torch.LongTensor(targets)
    return tensors, targets

def collate_fn_cnn(batch):
    tensors, targets = [], []

    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    tensors = pad_sequence_cnn(tensors)
    targets = torch.stack(targets)

    return tensors, targets