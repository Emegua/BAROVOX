import pandas as pd
import soundfile as sf
import numpy as np
import os, pdb, sys, glob
from pathlib import Path
from argparse import ArgumentParser

class Config:
    def __init__(self, args) -> None:
        self.parser = ArgumentParser(description='Parameters.')
        self.parser.add_argument('--csvPath', type=str, default="D:\\aicps\\cca\\Dataset_generator\\sampleAudio\\rpi\\pressureData\\2kvoices\\",help="Path for the audio to be processed")
        self.parser.add_argument('--wavPath', type=str, default="D:\\aicps\\cca\\Dataset_generator\\sampleAudio\\rpi\\pressureWav\\2kvoices\\", help="Sampling rate of the audio")
        self.parser.add_argument('--sr',  type=int, default=0)
        self.parser.add_argument('--row1', type=str, default="")
        self.parser.add_argument('--row2', type=str, default="1000")
        self.parser.add_argument('--skipRow', type=int, default=2)
        self.parser.add_argument('--norm', action="store_true")

        args_parsed = self.parser.parse_args(args)
        for arg_name in vars(args_parsed):
            self.__dict__[arg_name] = getattr(args_parsed, arg_name)

def csv2wav(cfg):
    for i in glob.glob((cfg.csvPath+"\\*.csv")):
        '''structure of the csv file. 
        time, 9520551  // duration
        sample, pressure
        0, -0.008
        1, 0.000
        '''
        outputFile = cfg.wavPath + Path(i).stem + ".wav"
        try:
            df = pd.read_csv(i, header=None, low_memory=False)
        except:
            continue

        #determine sampling rate:
        duration = int((df.iloc[0].tolist()[1])) # microsecond
        numberOfSamples = int(df[0].tolist()[-1]) 
        data = df[1].values[2:]
        sr = cfg.sr
        if cfg.sr == 0:
            sr = numberOfSamples*1000000//duration
        # write to a wav file

        data_norm = np.array(data,dtype=float)
        if cfg.norm:
            data_norm /= np.max(np.abs(data_norm))
        try: 
            sf.write(outputFile, data_norm, sr)
        except:
            pass

if __name__=="__main__":
    cfg = Config(sys.argv[1:])
    csv2wav(cfg)