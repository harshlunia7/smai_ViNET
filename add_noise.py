import librosa
import numpy as np
import math
import soundfile as sf
signal, sr = librosa.load("./sample_audio.wav")
RMS_signal=np.sqrt(np.mean(signal**2))
PSNR_target = -30
RMS_noise = np.sqrt(RMS_signal**2/(10**(PSNR_target/10)))
noise=np.random.normal(0, RMS_noise, signal.shape[0])
signal_noise = signal+noise
# Write out audio as 24bit PCM WAV
sf.write('stereo_file1.wav', signal_noise, sr, 'PCM_24')