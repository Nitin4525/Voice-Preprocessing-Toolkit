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
