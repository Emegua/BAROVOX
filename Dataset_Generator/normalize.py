import soundfile as sf
# import librosa
from ctypes import *
import glob
import subprocess, sys, time, os
import pandas as pd
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from pydub import silence
from pydub import AudioSegment, effects 
from scipy.io.wavfile import read, write
from joblib import Parallel, delayed
import multiprocessing

class Config:
    def __init__(self, args) -> None:
        self.parser = ArgumentParser(description='Parameters.')
        self.parser.add_argument('--s_path', type=str, default="dataset\\csv\\",help="Path for the audio to be processed")
        self.parser.add_argument('--w_path', type=str, default="dataset\\pressureWav\\", help="")
        self.parser.add_argument('--wsr_path', type=str, default="dataset\\pressureWavNoSilent\\", help="")

        self.parser.add_argument('--norm', action="store_true")

        args_parsed = self.parser.parse_args(args)
        for arg_name in vars(args_parsed):
            self.__dict__[arg_name] = getattr(args_parsed, arg_name)

def removeSilence(file_name, wsr_folder, cfg):
    outputFile = wsr_folder + "\\" +  Path(file_name).stem + ".wav"

    # make the audio in pydub audio segment format
    audio = AudioSegment.from_wav(file_name)
    
    #we may twick the silence_thresh value.
    audio_chunks = silence.split_on_silence(audio, min_silence_len=100, silence_thresh=-36, keep_silence = False)

    combined = AudioSegment.empty()
    for chunk in audio_chunks:
        combined += chunk
    combined.export(outputFile, format = "wav")

def csv2wav(file_name, w_folder, wsr_folder, cfg):
    # import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    outputFile = w_folder + "\\" +  Path(file_name).stem + ".wav"
    try:
        df = pd.read_csv(file_name, header=None, low_memory=False)
    except:
        pass

    #determine sampling rate:
    duration = int((df.iloc[0].tolist()[1])) # microsecond
    numberOfSamples = int(df[0].tolist()[-1]) 
    data = df[1].values[2:]

    sr = numberOfSamples*1000000//duration
    # write to a wav file

    data_norm = np.array(data,dtype=float)
    if cfg.norm:
        data_norm /= np.max(np.abs(data_norm))
    try: 
        sf.write(str(Path(outputFile)), data_norm, sr)
    except:
        pass
    removeSilence(str(Path(outputFile)), wsr_folder, cfg)

def normalize(file_name, d_path):
    def match_target_amplitude(sound, target_dBFS):
        change_in_dBFS = -1 - sound.dBFS
        return sound.apply_gain(change_in_dBFS)
    #import pdb; pdb.set_trace()
    sound = AudioSegment.from_file(file_name, "wav")
    normalized_sound = match_target_amplitude(sound, -3)
    outFileName =  d_path + "\\" +  Path(file_name).stem + ".wav"
    normalized_sound.export(outFileName, format="wav")


'''if __name__ == '__main__':
    for s_folder in glob.glob(os.path.join(str(Path(cfg.s_path)) + '\\*')):
        s_folder_stem = Path(s_folder).stem
        d_folder = f"{str(Path(cfg.d_path))}\\{s_folder_stem}"
        
        if not os.path.exists(d_folder):
            os.mkdir(d_folder)
        for file_name in glob.glob(os.path.join(s_folder + '\\*.wav')):
            normalize(file_name, d_folder)'''

def gen_dataset(cfg):
    for s_folder in glob.glob(os.path.join(cfg.s_path + '\\*')):
        s_folder_stem = Path(s_folder).stem
        w_folder = f"{cfg.w_path}\\{s_folder_stem}"
        wsr_folder = f"{cfg.wsr_path}\\{s_folder_stem}"
        if not os.path.exists(w_folder):
            os.mkdir(w_folder)
        if not os.path.exists(wsr_folder):
            os.mkdir(wsr_folder)
        
        num_cores = multiprocessing.cpu_count()-4
        Parallel(n_jobs=num_cores-4)(delayed(csv2wav)(file_name, w_folder, wsr_folder, cfg) for file_name in glob.glob(os.path.join(s_folder + '\\*.csv')))
        #[csv2wav(file_name, w_folder, wsr_folder, cfg) for file_name in glob.glob(os.path.join(s_folder + '\\*.csv'))]

if __name__ == '__main__':
    cfg = Config(sys.argv[1:])
    gen_dataset(cfg)