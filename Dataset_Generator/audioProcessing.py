import argparse, os, sys, scipy
from scipy.fft import fft
import matplotlib.pyplot as plt
import configparser as CP
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
import soundfile as sf
import librosa, pdb, glob
import librosa.display
from scipy.fft import fft

n_fft = 64
win_length = n_fft
hop_length = win_length//4

class Config:
    def __init__(self, args) -> None:
        self.parser = ArgumentParser(description='Parameters.')
        self.parser.add_argument('--audioPath', type=str, help="Path for the audio to be processed")
        self.parser.add_argument('--audioSR', default=2000, type=int, help="Sampling rate of the audio")
        self.parser.add_argument('--refAudioPath', default="", type=str, help="Reference audio path")
        self.parser.add_argument('--refAudioSR', type=int, help="Sampling rate of the refernce audio")
        self.parser.add_argument('--saveFilePath', type=str, default="sampleAudio/rpi/FFTPressureWav/")
        self.parser.add_argument('--lowerFreqRange', type=int, default=0)
        self.parser.add_argument('--upperFreqRange', type=int, default=4000)
        self.parser.add_argument('--scale', type=int, default=1)
        self.parser.add_argument('--mode', type=int, default=1)
        self.parser.add_argument('--n_fft', type=int, default=512)
        self.parser.add_argument('--hop_length', type=int, default=8)
        self.parser.add_argument('--win_length', type=int, default=64)
        self.parser.add_argument('--freq_low', type=int, default=0)
        self.parser.add_argument('--freq_high', type=int, default=1000)
        self.parser.add_argument('--cutoffFreq', type=int, default=30)
        self.parser.add_argument('--procType', type=str, default="spectWithMargin")


        args_parsed = self.parser.parse_args(args)
        for arg_name in vars(args_parsed):
            self.__dict__[arg_name] = getattr(args_parsed, arg_name)
       
def plotFFTBatch(cfg):
    folderNamePrefix = Path(cfg.audioPath).stem
    folderName = str(Path(cfg.saveFilePath + folderNamePrefix).resolve())
    if not os.path.exists(folderName):
        os.mkdir(folderName)
    '''if cfg.refAudioPath != "":
        audioPaths = [[x, cfg.refAudioPath+Path(x).stem+".wav"] for x in glob.glob(cfg.audioPath+"*.wav")]
    else:
        [[x] for x in glob.glob(cfg.audioPath+"*.wav")]'''
    for audioPath in glob.glob((cfg.audioPath+"*.wav")):
        fileNamePrefix = Path(audioPath).stem
        audio, _ = librosa.load(str(Path(audioPath)), sr=cfg.audioSR)
        genFFT(audio, cfg.audioSR, Path(cfg.audioPath).stem, cfg.freq_low, cfg.freq_high)
        try:
            if cfg.refAudioPath != "":
                refAudio, _ = librosa.load(cfg.refAudioPath + fileNamePrefix +".wav", sr=cfg.refAudioSR)
                genFFT(refAudio, cfg.refAudioSR, "Ref_" + fileNamePrefix, cfg.freq_low, cfg.freq_high)
        except:
            continue
        plt.legend()
        plt.title(f"FFT graph: Power vs Frequency")
        plt.savefig(folderName + '/FFT_' + fileNamePrefix + '.png')
        plt.close()

def plotFFT(audio, refAudio, folderName, fileNamePrefix, cfg):
    
    genFFT(refAudio, cfg.refAudioSR, Path(cfg.refAudioPath).stem, cfg.freq_low, cfg.freq_high)
    genFFT(audio, cfg.audioSR, Path(cfg.audioPath).stem, cfg.freq_low, cfg.freq_high)
    plt.legend()
    plt.title(f"FFT graph: Power vs Frequency")
    plt.savefig(folderName + '/FFT_' + fileNamePrefix + '.png')
    plt.close()
    pass

def genFFT(audio, sampling_rate,tag, lower=0, upper=1000):
    n= len(audio)
    T = 1/sampling_rate
    yf = fft(audio)
    xf = np.linspace(0.0, 1.0/(2.0*T), n//2)
    power = np.abs(yf[0:n//2])**2/n
    plt.plot(xf, power, label=tag)
    plt.grid()
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.xlim([lower, upper])
    #plt.ylim([0, 1])

def source_expand_modified(audioSTFT,refAudioSTFT,cfg, folderName, fileNamePrefix, saveFile=True):
    
    ImgName = "SourceExapand_" + fileNamePrefix 
    highPassedAudioSTFT = np.copy(audioSTFT)     
    expandedAudio = np.copy(audioSTFT)
    
    #Recover signals lost due to aliasing within the Nyquist frequency (0-500 HZ)
    audioNyquistSTFT = audioSTFT.shape[0]
    for i in range(2, audioNyquistSTFT):
        harmonicNumber = (audioSTFT.shape[0]-1)//i
        freqComponent = highPassedAudioSTFT[i, :] #checkThis, we might need to scale this down
        for k in range(1, harmonicNumber):
            expandedAudio[i + k*i] = expandedAudio[i + k*i] + freqComponent


    fig, ax = plt.subplots(nrows=3)
    rp = max(np.max(np.abs(audioSTFT)), max(np.max(np.abs(refAudioSTFT)), np.max(np.abs(expandedAudio))))
    audio_dB = librosa.power_to_db(audioSTFT, ref=rp)
    img = librosa.display.specshow(audio_dB, x_axis='time',
                        y_axis='log', sr=cfg.audioSR, n_fft=cfg.n_fft,  
                        ax=ax[0])
    expandedAudio_dB = librosa.power_to_db(expandedAudio, ref=rp)
    
    img = librosa.display.specshow(expandedAudio_dB, x_axis='time',
                        y_axis='log', sr=int(cfg.audioSR*cfg.scale), n_fft=cfg.n_fft*cfg.scale, fmax=int(4000), ax=ax[1])
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax[0].set(title='original')
    ax[1].set(title='expanded')
    if refAudioSTFT.any() != None:
        refAudio_dB = librosa.power_to_db(refAudioSTFT, ref=rp)
        img = librosa.display.specshow(refAudio_dB, x_axis='time', 
                            y_axis='log', sr=cfg.refAudioSR, n_fft=cfg.n_fft*cfg.scale,
                            fmax=int(4000), ax=ax[2])
        ax[2].set(title='reference')

    plt.savefig(Path(folderName + "/" + ImgName + ".png"))
    plt.close()
    expandedAudioSource = librosa.istft(expandedAudio, n_fft=(cfg.n_fft*cfg.scale))
    sf.write(folderName + "/" + ImgName + ".wav", expandedAudioSource, cfg.audioSR*cfg.scale)

def source_expand(audioSTFT, refAudioSTFT, cfg, folderName, fileNamePrefix, saveFile=True):
    ImgName = "SourceExapand_" + fileNamePrefix 
    highPassedAudioSTFT = np.copy(audioSTFT)     
    #highPassedAudioSTFT[:(cfg.cutoffFreq//cfg.n_fft), :] = np.zeros(((cfg.cutoffFreq//cfg.n_fft), audioSTFT.shape[1]))
    expandedAudio =  highPassedAudioSTFT # np.concatenate(tuple([highPassedAudioSTFT] + [np.zeros((cfg.n_fft//2, highPassedAudioSTFT.shape[1]))]*int(cfg.scale - 1)))
    
    #Recover signals lost due to aliasing within the Nyquist frequency (0-500 HZ)
    audioNyquistSTFT = audioSTFT.shape[0]//2+1
    for i in range(1, audioNyquistSTFT):
        harmonicNumber = (audioSTFT.shape[0]-1)//i
        freqComponent = highPassedAudioSTFT[i, :] #checkThis, we might need to scale this down
        for k in range(1, harmonicNumber):
            expandedAudio[i + k*i] = highPassedAudioSTFT[i+ k*i] + expandedAudio[i + k*i] + freqComponent

    fig, ax = plt.subplots(nrows=3)
    rp = max(np.max(np.abs(audioSTFT)), max(np.max(np.abs(refAudioSTFT)), np.max(np.abs(expandedAudio))))
    audio_dB = librosa.power_to_db(audioSTFT, ref=rp)
    img = librosa.display.specshow(audio_dB, x_axis='time',
                        y_axis='log', sr=cfg.audioSR, n_fft=cfg.n_fft,  
                        ax=ax[0])
    expandedAudio_dB = librosa.power_to_db(expandedAudio, ref=rp)
    
    img = librosa.display.specshow(expandedAudio_dB, x_axis='time',
                        y_axis='log', sr=int(cfg.audioSR*cfg.scale), n_fft=cfg.n_fft*cfg.scale, fmax=int(4000), ax=ax[1])
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax[0].set(title='original')
    ax[1].set(title='expanded')
    if refAudioSTFT.any() != None:
        refAudio_dB = librosa.power_to_db(refAudioSTFT, ref=rp)
        img = librosa.display.specshow(refAudio_dB, x_axis='time', 
                            y_axis='log', sr=cfg.refAudioSR, n_fft=cfg.n_fft*cfg.scale,
                            fmax=int(cfg.refAudioSR//2), ax=ax[2])
        ax[2].set(title='reference')

    plt.savefig(Path(folderName + "/" + ImgName + ".png"))

    pdb.set_trace()
    expandedAudioSource = librosa.istft(expandedAudio, n_fft=(cfg.n_fft*cfg.scale))
    sf.write(folderName + "/" + ImgName + ".wav", expandedAudioSource, cfg.audioSR*cfg.scale)


def separteHarmonicPercussive(audioSTFT, cfg, folderName, fileNamePrefix,saveFile=False):
    harm, perc = librosa.decompose.hpss(audioSTFT)
    harm_m8, perc_m8 = librosa.decompose.hpss(audioSTFT, margin=8)
    harm_m2, perc_m2 = librosa.decompose.hpss(audioSTFT, margin=2)

    nyquist_rate = cfg.audioSR / 2
    high_pass_cutoff = 32  # Hz
    order = 5
    b, a = scipy.signal.butter(order, high_pass_cutoff / nyquist_rate, btype='high')
    harm_filtered = scipy.signal.filtfilt(b, a, harm)

    audio_filtered = harm_m8 + 10*perc_m2

    audio_filtered = librosa.istft(audio_filtered)
    if saveFile:
        audio_harm = librosa.istft(harm_m8,   n_fft=cfg.n_fft)
        audio_perc = librosa.istft(10*perc_m2,   n_fft=cfg.n_fft)
        sf.write(cfg.audioPath[:-4] + "_filtered.wav", audio_filtered, cfg.audioSR)
        sf.write(cfg.audioPath[:-4] + "_harm.wav", audio_harm, cfg.audioSR)
        sf.write(cfg.audioPath[:-4] + "_perc.wav", audio_perc, cfg.audioSR)

    return audio_filtered

def reconstructAudio(audio, cfg, folderName, fileNamePrefix, saveFile=False):
    S = np.abs(librosa.stft(audio))

    # Estimate the frequency resolution of the original spectrogram
    f_resolution = librosa.core.spectrum.estimate_tuning(S=S, sr=cfg.audioSR)

    # Correct the aliased frequency components in the spectrogram
    S_corrected = librosa.core.spectrum.reconstruct_harmonics(S, cfg.audioSR, f_resolution=f_resolution)

    # Convert the corrected spectrogram back to the time domain
    y_corrected = librosa.istft(S_corrected)

    # Save the corrected audio
    sf.write(cfg.audioPath[:-4] + "_recon.wav", y_corrected, cfg.audioSR)
    
def plotSpectrogramWithMargin(audioSTFT, refAudioSTFT, folderName, fileNamePrefix,cfg=None):
    ImgName = "SpectWithMargin_" + fileNamePrefix 
    audioSTFT_harmonic, audioSTFT_percussive = librosa.decompose.hpss(audioSTFT)
    # Let's compute separations for a few different margins and compare the results below
    audioSTFT_harmonic2, audioSTFT_percussive2 = librosa.decompose.hpss(audioSTFT, margin=2)
    audioSTFT_harmonic4, audioSTFT_percussive4 = librosa.decompose.hpss(audioSTFT, margin=4)
    audioSTFT_harmonic8, audioSTFT_percussive8 = librosa.decompose.hpss(audioSTFT, margin=8)
    audioSTFT_harmonic20, audioSTFT_percussive20 = librosa.decompose.hpss(audioSTFT,  margin=20)
    
    refAudioSTFT_harmonic, refAudioSTFT_percussive = librosa.decompose.hpss(refAudioSTFT)
    ImgName = ImgName + "_" + Path(cfg.refAudioPath).stem
    n_rows=6
    rp = max(np.max(np.abs(audioSTFT)), np.max(np.abs(refAudioSTFT)))
    
    fig, ax = plt.subplots(nrows=n_rows, ncols=2, sharex=True, sharey=True, figsize=(20, 20))
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(audioSTFT_harmonic), ref=rp),
                            y_axis='log', sr=cfg.audioSR, n_fft=cfg.n_fft,  x_axis='time', ax=ax[0, 0])
    ax[0, 0].set(title='Harmonic')

    librosa.display.specshow(librosa.amplitude_to_db(np.abs(audioSTFT_percussive), ref=rp),
                            y_axis='log', sr=cfg.audioSR, n_fft=cfg.n_fft,  x_axis='time', ax=ax[0, 1])
    ax[0, 1].set(title='Percussive')

    librosa.display.specshow(librosa.amplitude_to_db(np.abs(audioSTFT_harmonic2), ref=rp),
                            y_axis='log', sr=cfg.audioSR, n_fft=cfg.n_fft,  x_axis='time', ax=ax[1, 0])

    librosa.display.specshow(librosa.amplitude_to_db(np.abs(audioSTFT_percussive2), ref=rp),
                            y_axis='log', sr=cfg.audioSR, n_fft=cfg.n_fft,  x_axis='time', ax=ax[1, 1])

    librosa.display.specshow(librosa.amplitude_to_db(np.abs(audioSTFT_harmonic4), ref=rp),
                            y_axis='log', sr=cfg.audioSR, n_fft=cfg.n_fft,  x_axis='time', ax=ax[2, 0])

    librosa.display.specshow(librosa.amplitude_to_db(np.abs(audioSTFT_percussive4), ref=rp),
                            y_axis='log', sr=cfg.audioSR, n_fft=cfg.n_fft,  x_axis='time', ax=ax[2, 1])

    librosa.display.specshow(librosa.amplitude_to_db(np.abs(audioSTFT_harmonic8), ref=rp),
                            y_axis='log', sr=cfg.audioSR, n_fft=cfg.n_fft,  x_axis='time', ax=ax[3, 0])

    librosa.display.specshow(librosa.amplitude_to_db(np.abs(audioSTFT_percussive8), ref=rp),
                            y_axis='log', sr=cfg.audioSR, n_fft=cfg.n_fft,  x_axis='time', ax=ax[3, 1])

    librosa.display.specshow(librosa.amplitude_to_db(np.abs(audioSTFT_harmonic20), ref=rp),
                            y_axis='log', sr=cfg.audioSR, n_fft=cfg.n_fft,  x_axis='time', ax=ax[4, 0])

    librosa.display.specshow(librosa.amplitude_to_db(np.abs(audioSTFT_percussive20), ref=rp),
                            y_axis='log', sr=cfg.audioSR, n_fft=cfg.n_fft,  x_axis='time', ax=ax[4, 1])

    librosa.display.specshow(librosa.amplitude_to_db(np.abs(refAudioSTFT_harmonic), ref=rp),
                            y_axis='log',  sr=cfg.refAudioSR, n_fft=cfg.n_fft*cfg.scale,  
                             x_axis='time', ax=ax[5, 0])

    librosa.display.specshow(librosa.amplitude_to_db(np.abs(refAudioSTFT_percussive), ref=rp),
                            y_axis='log',  sr=cfg.refAudioSR, n_fft=cfg.n_fft*cfg.scale,  
                             x_axis='time', ax=ax[5, 1])
        
    for i in range(n_rows-1):
        ax[i, 0].set(ylabel='margin={:d}'.format(2**i))
        ax[i, 0].label_outer()
        ax[i, 1].label_outer()
    ax[n_rows-1, 0].set(ylabel='Reference')
    ax[n_rows-1, 0].label_outer()
    ax[n_rows-1, 1].label_outer()

    plt.savefig(Path(folderName + "/" + ImgName + ".png"))
    plt.close()

def melSprctrogram(audio, refAudio=None, cfg=None):
    S = librosa.feature.melspectrogram(y=audio,  n_mels=20,n_fft=cfg.n_fft)
    ImgName = Path(cfg.audioPath).stem
    n_cols = 1
    if refAudio != None:
        refS = librosa.feature.melspectrogram(y=refAudio, sr=cfg.refAudioSR, n_mels=20,n_fft=cfg.n_fft)
        ImgName = ImgName + "_" + Path(cfg.refAudioPath).stem
        n_cols = 2
    fig, ax = plt.subplots(ncols=n_cols)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    if refAudio != None:
        img = librosa.display.specshow(S_dB, x_axis='time',
                            y_axis='mel', 
                            ax=ax[0])
        refS_dB = librosa.power_to_db(refS, ref=np.max)
        img = librosa.display.specshow(refS_dB, x_axis='time',
                            y_axis='mel',  sr=cfg.refAudioSR, n_fft=cfg.n_fft*cfg.scale,
                            ax=ax[1])
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax[0].set(title='Mel')
        ax[1].set(title='Ref Mel')
    else: 
        img = librosa.display.specshow(S_dB, x_axis='time',
                            y_axis='mel', 
                            ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title='Mel')
    plt.savefig(Path(cfg.saveFilePath + "/" + cfg.procType + "_" + ImgName + ".png"))

## REf: https://librosa.org/doc/latest/auto_examples/plot_hprss.html#sphx-glr-auto-examples-plot-hprss-py
def plotSpectrogram(audioSTFT, refAudioSTFT, folderName, fileNamePrefix, cfg=None):
    ImgName = "SpectWithMargin_" + fileNamePrefix 
    audioSTFT_harmonic, audioSTFT_percussive = librosa.decompose.hpss(audioSTFT)
    refAudioSTFT_harmonic, refAudioSTFT_percussive = librosa.decompose.hpss(refAudioSTFT)

    rp = max(np.max(np.abs(audioSTFT)), np.max(np.abs(refAudioSTFT)))
    fig, ax = plt.subplots(nrows=3,ncols=2, sharex=True, sharey=True, figsize=(20, 20))

    img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(audioSTFT), ref=rp),
                            y_axis='log', sr=cfg.audioSR,  n_fft=cfg.n_fft,  x_axis='time', ax=ax[0, 0])
    ax[0, 0].set(title='Full spectrogram')

    librosa.display.specshow(librosa.amplitude_to_db(np.abs(audioSTFT_harmonic), ref=rp),
                            y_axis='log', sr=cfg.audioSR, n_fft=cfg.n_fft,  x_axis='time', ax=ax[1, 0])
    ax[1, 0].set(title='Harmonic spectrogram')

    librosa.display.specshow(librosa.amplitude_to_db(np.abs(audioSTFT_percussive), ref=rp),
                            y_axis='log', sr=cfg.audioSR, n_fft=cfg.n_fft,   x_axis='time', ax=ax[2, 0])
    ax[2, 0].set(title='Percussive spectrogram')

    librosa.display.specshow(librosa.amplitude_to_db(np.abs(refAudioSTFT), ref=rp), y_axis='log', 
                             sr=cfg.refAudioSR,   x_axis='time', ax=ax[0, 1])
    ax[0, 1].set(title='Full spectrogram')
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(refAudioSTFT_harmonic), ref=rp), y_axis='log', 
                             sr=cfg.refAudioSR,   x_axis='time', ax=ax[1, 1])
    ax[1, 1].set(title='Reference Harmonic spectrogram')
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(refAudioSTFT_percussive), ref=rp),y_axis='log', 
                             sr=cfg.refAudioSR,   x_axis='time', ax=ax[2, 1])
    ax[2, 1].set(title='Reference Percussive spectrogram')
    fig.colorbar(img, ax=ax)

    
    plt.savefig(Path(folderName + "/" + ImgName + ".png"))


def plotPerceptualWeighting(audio, refAudio, cfg):
    audioSTFT = librosa.stft(audio, n_fft=cfg.n_fft)
    refAudioSTFT = librosa.stft(refAudio, n_fft=cfg.n_fft*cfg.scale)
    freq = librosa.fft_frequencies(sr=cfg.audioSR, n_fft=cfg.n_fft)
    refFreq = librosa.fft_frequencies(sr=cfg.refAudioSR, n_fft=cfg.n_fft*cfg.scale)

    audioChromagram = librosa.perceptual_weighting(abs(audioSTFT)**2, freq)
    refAudioChromagram = librosa.perceptual_weighting(abs(refAudioSTFT)**2, refFreq)
    
    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(20, 20))

    img = librosa.display.specshow(audioChromagram, x_axis='time', y_axis='log', sr=cfg.audioSR, cmap='coolwarm', ax=ax[0])
    img = librosa.display.specshow(refAudioChromagram, x_axis='time', y_axis='log',  sr=cfg.refAudioSR, n_fft=cfg.n_fft*cfg.scale, cmap='coolwarm', ax=ax[1])

    ax[0].set(title='PerceptualWeighting')
    ax[1].set(title='RefPerceptualWeighting')
    fig.colorbar(img, ax=ax)    
    plt.savefig(Path(cfg.saveFilePath + "/" + "PerceptualWeighting" + ".png"))
    print("done")

def plotSpectralContrast(audioSTFT, refAudioSTFT,folderName, fileNamePrefix, cfg):
    pdb.set_trace()
    ImgName = "SpectContrast_" + fileNamePrefix 
    audioContrast = librosa.feature.spectral_contrast(S=np.abs(audioSTFT), sr=cfg.audioSR, n_fft=cfg.n_fft)
    refAudioContrast = librosa.feature.spectral_contrast(S=np.abs(refAudioSTFT), sr=cfg.refAudioSR, n_fft=cfg.n_fft*(cfg.refAudioSR//cfg.audioSR))

    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(20, 20))

    img = librosa.display.specshow(audioContrast, x_axis='time',  sr=cfg.audioSR, 
                                    n_fft=cfg.n_fft,ax=ax[0])
    img = librosa.display.specshow(refAudioContrast, x_axis='time',   sr=cfg.refAudioSR, 
                                    n_fft=cfg.n_fft*cfg.scale,ax=ax[1])

    ax[0].set(title='Contrast')
    ax[1].set(title='RefContrast')
    fig.colorbar(img, ax=ax)    
    plt.savefig(Path(folderName + "/" + ImgName + ".png"))
    plt.close()

def plotChromaSTFT(audio, refAudio, cfg):
    audioChromagram = librosa.feature.chroma_stft(audio, sr=cfg.audioSR, n_fft=cfg.n_fft)
    refAudioChromagram = librosa.feature.chroma_stft(refAudio, sr=cfg.refAudioSR, n_fft=cfg.n_fft*cfg.scale)

    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(20, 20))

    img = librosa.display.specshow(audioChromagram, x_axis='time', y_axis='log', sr=cfg.audioSR, cmap='coolwarm', ax=ax[0])
    img = librosa.display.specshow(refAudioChromagram, x_axis='time', y_axis='log',  sr=cfg.refAudioSR, n_fft=cfg.n_fft*cfg.scale, cmap='coolwarm', ax=ax[1])

    ax[0].set(title='Chroma')
    ax[1].set(title='RefChroma')
    fig.colorbar(img, ax=ax)    
    plt.savefig(Path(cfg.saveFilePath + "/" + "chromagram" + ".png"))
    print("done")

def processAudio(cfg):    
    audio, _ = librosa.load(str(Path(cfg.audioPath)), sr=cfg.audioSR)
    
    audioSTFT = librosa.stft(audio, n_fft=cfg.n_fft)


    refAudio = None
    refAudioSTFT = None
    if cfg.refAudioPath:
        refAudio, _ = librosa.load(str(Path(cfg.refAudioPath)), sr=cfg.refAudioSR)
        refAudioSTFT = librosa.stft(refAudio, n_fft=cfg.n_fft*cfg.scale)
    #pdb.set_trace()
    if cfg.procType == "spectWithMargin":
        plotSpectrogramWithMargin(audioSTFT, refAudioSTFT, cfg)
    elif cfg.procType == "mel":
        melSprctrogram(audio, refAudio, cfg)
    elif cfg.procType == "spect":
        plotSpectrogram(audioSTFT, refAudioSTFT, cfg)
    elif cfg.procType == "spectCont":
        plotSpectralContrast(audioSTFT, refAudioSTFT, cfg)
    elif cfg.procType == "hpss":
        separteHarmonicPercussive(audioSTFT, cfg, True)
    elif cfg.procType == "hpssSpectWithMargin":
        audioSTFT = librosa.stft(separteHarmonicPercussive(audioSTFT, cfg), n_fft=cfg.n_fft)
        plotSpectrogramWithMargin(audioSTFT, refAudioSTFT, cfg)
    elif cfg.procType == "recon":
        reconstructAudio(audio, cfg, True)
    elif cfg.procType == "exp":
        source_expand(audioSTFT, cfg, refAudioSTFT, True)
    elif cfg.procType == "chroma":
        plotChromaSTFT(audio, refAudio, cfg)
    pass

def analyzeAudios(audioPath, refAudioPath, folderName, cfg):
    fileNamePrefix = Path(audioPath).stem + "_" + Path(refAudioPath).stem

    audio, _ = librosa.load(str(Path(audioPath)), sr=cfg.audioSR)
    audioSTFT = librosa.stft(audio, n_fft=cfg.n_fft)
    refAudio, _ = librosa.load(str(Path(refAudioPath)), sr=cfg.refAudioSR)
    refAudioSTFT = librosa.stft(refAudio, n_fft=cfg.n_fft*cfg.scale)

    #folderName = str(Path(folderName + "/" + fileNamePrefix).resolve())
    if not os.path.exists(folderName):
        os.mkdir(folderName)
    #plot FFT 
    if cfg.mode & 1:
        plotFFT(audio, refAudio, folderName, fileNamePrefix, cfg)
    if cfg.mode & 2:
        plotSpectrogram(audioSTFT, refAudioSTFT, folderName, fileNamePrefix, cfg)
    if cfg.mode & 4:
        plotSpectrogramWithMargin(audioSTFT, refAudioSTFT, folderName, fileNamePrefix, cfg)
    #plotSpectralContrast(audioSTFT, refAudioSTFT, folderName, fileNamePrefix, cfg)
    if cfg.mode & 8:
        source_expand_modified(audioSTFT, refAudioSTFT, cfg, folderName, fileNamePrefix, True)
    if cfg.mode & 8:
        plotFFTBatch(cfg)

def analyzeAudiosInBatch(cfg):
    fileNamePrefix = Path(cfg.audioPath).stem
    folderName = str(Path(cfg.saveFilePath + fileNamePrefix + "/").resolve())
    if not os.path.exists(folderName):
        os.mkdir(folderName)
    for audioPath in glob.glob((cfg.audioPath+"*.wav")):
        fileNamePrefix = Path(audioPath).stem
        refAudioPath = ""
        if cfg.refAudioPath != "":
            refAudiopath = cfg.refAudioPath + fileNamePrefix +".wav"
        analyzeAudios(audioPath, refAudiopath, folderName, cfg=cfg)
        """if cfg.mode & 1:
            plotFFTBatch(audioPath, refAudiopath, cfg)
        if cfg.mode & 4:
            plotSpectrogramWithMargin(audioSTFT, refAudioSTFT, folderName, fileNamePrefix, cfg)"""
    pass

if __name__=="__main__":
    cfg = Config(sys.argv[1:])
    if cfg.mode & 16:
        
        plotFFTBatch(cfg)
        #analyzeAudiosInBatch(cfg)
    else:
        analyzeAudios(cfg.audioPath, cfg.refAudioPath, cfg.saveFilePath, cfg)