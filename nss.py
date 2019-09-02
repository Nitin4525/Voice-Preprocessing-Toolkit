import numpy as np
import librosa
from scipy.io import wavfile

filepath = 'p232_007.wav'
wav, sr = librosa.load(filepath, 16000)

noise_estimated = wav[:int(1000 * sr / 1000)]

fft_wav = np.fft.fft(wav)
mag_fft_wav = np.abs(fft_wav)
phase_fft_wav = np.angle(fft_wav)

fft_noise = np.fft.fft(noise_estimated)
E_noise = np.sum(abs(fft_noise)) / len(noise_estimated)

mag_s = mag_fft_wav - E_noise
mag_s[mag_s < 0] = 0

fft_s = np.multiply(mag_s, np.exp(np.multiply(complex(0, 1), phase_fft_wav)))
enhance_res = np.real(np.fft.ifft(fft_s))

filename = filepath.split('.')[0] + '_nas_res.wav'
wavfile.write(filename, sr, enhance_res)
