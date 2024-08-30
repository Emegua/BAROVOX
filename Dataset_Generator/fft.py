import argparse, os, librosa
from librosa import display
from scipy.fft import fft
import matplotlib.pyplot as plt
import configparser as CP
import numpy as np


def plot(x, y, xlabel="Time", ylabel="Amplitude"):
    plt.figure()
    plt.stem(x, y, 'r',)
    plt.plot(x, y)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()

def fft_plot(audio, sampling_rate):
    n= len(audio)
    T = 1/sampling_rate
    yf = fft(audio)
    # yf = yf[~np.isnan(yf)]
    xf = np.linspace(0.0, 1.0/(2.0*T), n//2)
    # plt.plot(xf, 2.0/n*np.abs(yf[:n//2]))
    power = np.abs(yf[0:n//2])**2/n
    plt.plot(xf, power)
    plt.grid()
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.xlim([0, 700])
    

def fft_plot_combined(audio1, audio2, sampling_rate, label2):
    fig = plt.figure()
    
    ax = fig.add_subplot(1,1,1)
    fft_plot(audio1, sampling_rate, "Microphone")
    fft_plot(audio2, sampling_rate, label2)
    plt.grid()


# Just for example, you can visualize the FFT of sinWave using this function. 
def sinWave():
    samples=100
    f = 5
    x = np.arange(samples)
    y = np.sin(2*np.pi*f*(x/samples))
    fft_plot(y, samples)
    return x, y



def gen_fft(cfg):
    file_path  = cfg["file_path"]
    save_folder = cfg["save_folder_path"]
    sampling_rate = int(cfg["sampling_rate"])
    letters = list(cfg["letters"].strip("[, ]").split(", "))
    

    # If you want to visualize the fft plot
    for i in letters:
        mic, _ = librosa.load(file_path + f'mic/{i}_01.wav', sr=sampling_rate)
        sensor, _ = librosa.load(file_path + f'pressure/{i}.wav', sr=sampling_rate)
        normalized, _ = librosa.load(file_path + f'pressureNorm/{i}_norm.wav', sr=sampling_rate)

        fft_plot(mic, sampling_rate)
        plt.title(f"'{i}' Microphone Power vs Frequency")
        plt.savefig(save_folder + f"MicrophoneGraph{i}.jpg")
        plt.close()
        
        fft_plot(sensor, sampling_rate)
        plt.title(f"'{i}' Sensor Power vs Frequency")
        plt.savefig(save_folder + f"SensorGraph{i}.jpg")
        plt.close()
        
        fft_plot(normalized, sampling_rate)
        plt.title(f"'{i}' Normalized Sensor Power vs Frequency")
        plt.savefig(save_folder + f"NormalizedGraphs{i}.jpg")
        plt.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="fft.cfg")
    parser.add_argument("--cfg_str", type=str, default = "default" )
    args = parser.parse_args()
    cfgpath = os.path.join(os.path.dirname(__file__), args.cfg)
    assert os.path.exists(cfgpath), f"No configuration file as [{cfgpath}]"
    cfg = CP.ConfigParser()
    cfg._interpolation = CP.ExtendedInterpolation()
    cfg.read(cfgpath)
    
    gen_fft(cfg._sections[args.cfg_str])
    
    
