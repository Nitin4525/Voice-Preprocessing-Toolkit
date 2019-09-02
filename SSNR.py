import numpy as np


def SegSNR(wav_ref, wav_enhance, sr=16000, ep1=1e-10, ep2=1e-5):

    if len(wav_ref) == len(wav_enhance):
        pass
    else:
        print('SSNR: the length of wav is not equal!')
        minlenth = min(len(wav_ref), len(wav_enhance))
        wav_ref = wav_ref[:minlenth]
        wav_enhance = wav_enhance[:minlenth]

    framesize = int(20 * sr / 1000)
    # 直接抛弃语音结尾后边不足一帧的数据，因为一帧20ms，不足一帧的数据不影响结果
    framenum = int(len(wav_ref) / framesize)

    k = 0
    SegSNR = np.zeros(framenum)
    for i in range(framenum):
        wav_ref_frame = wav_ref[k:k + framesize]
        wav_enhance_frame = wav_enhance[k:k + framesize]
        SegSNR[i] = 10 * np.mean(np.log10(wav_ref_frame**2 /
                                          ((wav_ref_frame - wav_enhance_frame)**2 + ep1) + ep2))
        k = k + framesize
        # print(SegSNR)

    return np.mean(SegSNR)

'''
path = 'lenthBN'

for rootdir, subdir, files in os.walk(path):

    if len(files) == 0:
        continue

    filename = rootdir.split('\\')

    wav_clean, sr = librosa.load(
        filename[0] + os.sep + filename[1] + os.sep + filename[1] + '_clean.wav', 16000)

    wav_nss, sr = librosa.load(filename[0] +
                               os.sep +
                               filename[1] +
                               os.sep +
                               filename[1] +
                               '_nss.wav', 16000)

    wav_newnss, sr = librosa.load(
        filename[0] + os.sep + filename[1] + os.sep + filename[1] + '_newnss.wav', 16000)

    wav_kaelman, sr = librosa.load(
        filename[0] + os.sep + filename[1] + os.sep + filename[1] + '_kalman.wav', 16000)

    wav_weina, sr = librosa.load(
        filename[0] + os.sep + filename[1] + os.sep + filename[1] + '_weina.wav', 16000)

    wav_segan, sr = librosa.load(filename[0] +
                                 os.sep +
                                 filename[1] +
                                 os.sep +
                                 filename[1] +
                                 '_enhanced.wav', 16000)

    wav_noisy, sr = librosa.load(
        filename[0] + os.sep + filename[1] + os.sep + filename[1] + '_noisy.wav', 16000)
'''
