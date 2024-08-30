## Dataset Processing
We have 2 main scripts to process recorded pressure wav and convert it to audio. `csvtoway.py` and `audioProcessing.py`. `csvtoway.py`  is used for converting CSV files containing audio data into WAV format. `audioProcessing.py` script is used for processing recorded pressure sensor data and convert it to wav files and perform various operations on the converted audio data.


## Dependencies

- scipy
- matplotlib
- configparser
- numpy
- pathlib
- soundfile
- librosa
- glob
- pandas

### Configuration `csvtoway.py`
The script uses a `Config` class to handle command-line arguments. The arguments include:

- `--csvPath`: Path for the CSV file to be converted.
- `--wavPath`: Path where the converted WAV file will be saved.
- `--sr`: Sampling rate of the audio (default is 0).
- `--row1`: The first row to be read from the CSV file (default is an empty string).
- `--row2`: The last row to be read from the CSV file (default is "1000").
- `--skipRow`: Number of rows to skip at the beginning of the CSV file (default is 2).
- `--norm`: If set, the audio data will be normalized.

### Usage `csvtoway.py`

To use this script, you need to pass the necessary command-line arguments. Here is an example:

```bash
python csvtowav.py --csvPath /path/to/csv --wavPath /path/to/save/wav --sr 44100 --row1 0 --row2 1000 --skipRow 2 --norm
```


### Configuration `audioProcessing.py`

The script uses a `Config` class to handle command-line arguments. The arguments include:

- `--audioPath`: Path for the audio to be processed.
- `--audioSR`: Sampling rate of the audio (default is 2000).
- `--refAudioPath`: Reference audio path (default is an empty string).
- `--refAudioSR`: Sampling rate of the reference audio.
- `--saveFilePath`: Path where the processed audio will be saved (default is "sampleAudio/rpi/FFTPressureWav/").
- `--lowerFreqRange`: Lower frequency range for processing (default is 0).
- `--upperFreqRange`: Upper frequency range for processing (default is 4000).
- `--scale`: Scale factor for the audio (default is 1).
- `--mode`: Mode for processing (default is 1).
- `--n_fft`: FFT window size (default is 512).
- `--hop_length`: Hop length for FFT (default is 8).
- `--win_length`: Window length for FFT (default is 64).

## Usage

To use this script, you need to pass the necessary command-line arguments. Here is an example:

```bash
python audioProcessing.py --audioPath /path/to/audio --audioSR 44100 --refAudioPath /path/to/reference/audio --refAudioSR 44100 --saveFilePath /path/to/save/processed/audio
```