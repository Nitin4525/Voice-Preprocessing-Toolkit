# 非迭代维纳滤波的实现，照着一个github的MATLAB工程写的

import numpy as np
import scipy.signal as ss


def get_filter_arg(sr, framesize, nfft=8000):
    # 用于获取滤波器参数，返回ifilter的参数b，维度为(nfft,len(fc))，nfft默认为8000
    a = 1
    N = 4
    fc = np.array([50, 150, 250, 350, 450,
                   570, 700, 840, 1000, 1170,
                   1370, 1600, 1850, 2150, 2500,
                   2900, 3400])
    t = np.arange(1, framesize + 1)
    t1 = (framesize - t) / sr
    t = t / sr
    ERB = 24.7 + 0.108 * fc     # 此处包括后面的常数都是典型值
    b = 2 * np.pi * ERB * 1.019 * (-1)
    pt = []
    gt = []
    for i in range(len(fc)):
        temp_pt = a * (t**(N - 1)) * \
            np.exp(b[i] * t) * np.cos(2 * np.pi * fc[i] * t)
        temp_gt = a * (t1**(N - 1)) * \
            np.exp(b[i] * t1) * np.cos(2 * np.pi * fc[i] * t1)
        pt.append(temp_pt)
        gt.append(temp_gt)  # 注意数据类型，使用列表便于append，计算时转化为numpy矩阵
    pt = np.array(pt)
    gt = np.array(gt)

    Fpt = []
    Fgt = []
    for i in range(len(fc)):
        temp_Fpt = np.fft.fft(pt[i, :], nfft)
        temp_Fgt = np.fft.fft(gt[i, :], nfft)
        Fpt.append(temp_Fpt)
        Fgt.append(temp_Fgt)
    Fpt = np.array(Fpt)
    Fgt = np.array(Fgt)

    Fqt = []
    qt = []
    for i in range(len(fc)):
        temp_Fqt = Fpt[i, :] * Fgt[i, :]
        Fqt.append(temp_Fqt)
    Fqt = np.array(Fqt)

    for i in range(len(fc)):
        temp_qt = np.fft.ifft(Fqt[i, :])
        temp_qt = temp_qt / max(temp_qt)
        qt.append(temp_qt)
    qt = np.array(qt)
    return qt.T, fc


def wiener(wav, sr):

    framesize = int(10 * sr / 1000)
    qt, fc = get_filter_arg(sr, framesize)

    lenth = len(wav)
    framenum = lenth / framesize
    # 不足整帧的，需要进行补零操作
    if framenum == int(framenum):
        pass
    else:
        padding = framesize * (int(framenum) + 1) - lenth
        wav = np.pad(wav, (0, padding))

    xm = []
    u = 100     # 拉格朗日乘子，这个根据文章设置的
    for i in range(len(fc)):
        temp_xm = []
        En = 0
        # zf的大小为max(len(b),len(a))-1，a和b分别是ifilter参数
        zf = np.zeros(len(qt[i, :]) - 1)
        for j in range(5):  # 取前五帧作为噪声估计
            xmin, zf = ss.lfilter(
                qt[i, :], 1, wav[j * framesize:(j + 1) * framesize], zi=zf)
            En = En + np.sum(np.abs(xmin) ** 2)
        En = En / 5
        zf = np.zeros(len(qt[i, :]) - 1)
        for n in range(int(framenum)):
            xmin, zf = ss.lfilter(
                qt[i, :], 1, wav[n * framesize:(n + 1) * framesize], zi=zf)
            xmin = np.real(xmin)
            Esn = np.sum(np.abs(xmin) ** 2)
            Es = Esn - En
            k = Es / (Es + u * En)
            xim1 = xmin * k
            temp_xm.append(xim1)
        xm.append(temp_xm)
    xm = np.array(xm)
    xm = xm.reshape(17, xm.shape[1] * xm.shape[2])
    enhance = np.sum(xm, 0)
    return enhance / np.max(enhance)


'''
import librosa
wav, sr = librosa.load('test.wav', 16000)
framesize = int(10 * sr / 1000)
qt, fc = get_filter_arg(sr, framesize)

lenth = len(wav)
framenum = lenth / framesize
if framenum == int(framenum):
    pass
else:
    padding = framesize * (int(framenum) + 1) - lenth
    wav = np.pad(wav, (0, padding))

xm = []
u = 100
for i in range(len(fc)):
    temp_xm = []
    En = 0
    zf = np.zeros(16)
    for j in range(5):  # 取前五帧作为噪声估计
        xmin, zf = ss.lfilter(
            qt[i, :], 1, wav[j * framesize:(j + 1) * framesize], zi=zf)
        En = En + np.sum(np.abs(xmin)**2)

    En = En / 5
    zf = np.zeros(16)
    for n in range(int(framenum)):
        xmin, zf = ss.lfilter(
            qt[i, :], 1, wav[n * framesize:(n + 1) * framesize], zi=zf)
        xmin = np.real(xmin)
        Esn = np.sum(np.abs(xmin)**2)
        Es = Esn - En
        k = Es / (Es + u * En)
        xim1 = xmin * k
        temp_xm.append(xim1)
    xm.append(temp_xm)
xm = np.array(xm)
xm = xm.reshape(17, xm.shape[1] * xm.shape[2])
enhance = np.sum(xm, 0)
enhance = enhance / np.max(enhance)
'''
