import librosa
import numpy as np
import math
import soundfile as sf
import os
import glob

data_dir = "/ssd_scratch/cvit/rafaelgetto/AVAD/video_audio"
out_dir = "/ssd_scratch/cvit/rafaelgetto/AVAD/video_noise"
video_names = os.listdir(data_dir)
PSNR_target = -30

for v in video_names:
    print(v)
    audio_file = glob.glob(os.path.join(data_dir, v, "./*wav"))
    assert len(audio_file) == 1
    audio_file_name = audio_file[0].split("/")[-1]
    signal, sr = librosa.load(audio_file[0])
    RMS_signal = np.sqrt(np.mean(signal**2))
    RMS_noise = np.sqrt(RMS_signal**2 / (10 ** (PSNR_target / 10)))
    noise = np.random.normal(0, RMS_noise, signal.shape[0])
    signal_noise = signal + noise
    # Write out audio as 24bit PCM WAV
    os.makedirs(os.path.join(out_dir, v))
    sf.write(os.path.join(out_dir, v, audio_file_name), signal_noise, sr, "PCM_24")
