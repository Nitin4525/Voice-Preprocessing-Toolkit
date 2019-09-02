import numpy as np
import librosa
import math
from scipy.io import wavfile


def berouti1(SNR):
    if SNR >= -5.0 and SNR <= 20.0:
        alpha = 3 - SNR * 2 / 20
    elif SNR <= -5.0:
        alpha = 4
    else:
        alpha = 1
    return alpha


def berouti(SNR):
    if SNR >= -5.0 and SNR <= 20.0:
        alpha = 4 - SNR * 3 / 20
    elif SNR <= -5.0:
        alpha = 5
    else:
        alpha = 1
    return alpha


filepath = 'p232_007.wav'
wav, sr = librosa.load(filepath, 16000)

framesize = int(20 * sr / 1000)
if framesize % 2 == 1:
    framesize = framesize + 1
overlap = 50    # 帧叠百分比
framesize1 = int(framesize * overlap / 100)
framesize2 = framesize - framesize1

thres = 3   # vad中SNR的阈值
expnt = 2.0     # 幅度谱(1.0)or功率谱(2.0)
beta = 0.02    # 论文中提到的β
G = 0.9

hamwin = np.hamming(framesize)
winGain = framesize2 / np.sum(hamwin)     # 增益归一化

# 估计噪声幅度谱，设前5帧为噪声，大约100ms(实际在语音段中只占了60ms)
nfft = 2 * (2**int(math.ceil(math.log2(framesize))))
noise_mean = np.zeros(nfft)
j = 0
for k in range(5):
    noise_mean = noise_mean + \
        np.abs(np.fft.fft(np.multiply(hamwin, wav[j:j + framesize]), nfft))
    j = j + framesize
noise_mean = noise_mean / 5

# 谱减法处理语音
k = 0
wav_src = np.zeros(framesize1)
framenum = int(len(wav) / framesize2) - 1     # 如果最后语音尾部出现缺失就是这儿写错了
padding = framenum * framesize2 + framesize - len(wav)      # 在语音末尾补0，使之满足整帧长度
wav = np.pad(wav, (0, padding), 'constant')
wav_res = np.zeros((framenum+1) * framesize2)

for n in range(framenum + 1):
    wav_win = np.multiply(hamwin, wav[k:k + framesize])     # 最后一帧果然会出问题，还是补零吧。。。
    wav_win_fft = np.fft.fft(wav_win, nfft)
    wav_win_mag = np.abs(wav_win_fft)
    wav_win_phase = np.angle(wav_win_fft)
    SNRseg = 10 * math.log10((np.linalg.norm(wav_win_mag, 2)
                              ** 2) / (np.linalg.norm(noise_mean, 2)**2))
    if expnt == 1.0:
        alpha = berouti1(SNRseg)    # 也是论文里的alpha
    else:
        alpha = berouti(SNRseg)

    sub_speech = wav_win_mag**expnt - alpha * (noise_mean**expnt)   # 这儿用的是功率谱
    diffw = sub_speech - beta * (noise_mean**expnt)
    z = np.where(diffw < 0)
    if z[0].shape[0] >= 0:
        for i in range(z[0].shape[0]):
            sub_speech[z[0][i]] = beta * (noise_mean[z[0][i]]**expnt)
    if SNRseg <= thres:
        noise_temp = G * noise_mean**expnt + (1 - G) * wav_win_mag**expnt
        noise_mean = noise_temp**(1 / expnt)

    sub_speech[int(nfft / 2) + 2:nfft -
               1] = np.flipud(sub_speech[2:int(nfft / 2) - 1])
    wav_res_fft = np.multiply(sub_speech**(1 / expnt), np.exp(np.multiply(complex(0, 1), wav_win_phase)))
    wav_res_time = np.real(np.fft.ifft(wav_res_fft))
    wav_res[k:k + framesize2] = wav_src + wav_res_time[0:framesize1]
    k = k + framesize2

filename = filepath.split('.')[0] + '_newnas_res.wav'
wavfile.write(filename, sr, winGain * wav_res)
