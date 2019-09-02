import librosa
import numpy as np
import os

path = 'test'
wavlen = np.zeros(7)

for rootdir, subdir, files in os.walk(path):

    if len(files) == 0:
        continue

    filename = rootdir.split('\\')

    wav_clean, sr = librosa.load(
        filename[0] + os.sep + filename[1] + os.sep + filename[1] + '_clean.wav', 16000)
    wavlen[0] = len(wav_clean)

    wav_nss, sr = librosa.load(filename[0] +
                               os.sep +
                               filename[1] +
                               os.sep +
                               filename[1] +
                               '_nss.wav', 16000)
    wavlen[1] = len(wav_nss)

    wav_newnss, sr = librosa.load(
        filename[0] + os.sep + filename[1] + os.sep + filename[1] + '_newnss.wav', 16000)
    wavlen[2] = len(wav_newnss)

    wav_kaelman, sr = librosa.load(
        filename[0] + os.sep + filename[1] + os.sep + filename[1] + '_kalman.wav', 16000)
    wavlen[3] = len(wav_kaelman)

    wav_weina, sr = librosa.load(
        filename[0] + os.sep + filename[1] + os.sep + filename[1] + '_weina.wav', 16000)
    wavlen[4] = len(wav_weina)

    wav_segan, sr = librosa.load(filename[0] +
                                 os.sep +
                                 filename[1] +
                                 os.sep +
                                 filename[1] +
                                 '_enhanced.wav', 16000)
    wavlen[5] = len(wav_segan)

    wav_noisy, sr = librosa.load(
        filename[0] + os.sep + filename[1] + os.sep + filename[1] + '_noisy.wav', 16000)
    wavlen[6] = len(wav_noisy)

    min_len = np.min(wavlen)

    wav_clean = wav_clean[:int(min_len)]
    wav_nss = wav_nss[:int(min_len)]
    wav_newnss = wav_newnss[:int(min_len)]
    wav_kaelman = wav_kaelman[:int(min_len)]
    wav_weina = wav_weina[:int(min_len)]
    wav_segan = wav_segan[:int(min_len)]
    wav_noisy = wav_noisy[:int(min_len)]

    librosa.output.write_wav(
        'lenthBN' +
        os.sep +
        filename[1] +
        os.sep +
        filename[1] +
        '_clean.wav',
        wav_clean,
        16000)
    librosa.output.write_wav(
        'lenthBN' +
        os.sep +
        filename[1] +
        os.sep +
        filename[1] +
        '_nss.wav',
        wav_nss,
        16000)
    librosa.output.write_wav(
        'lenthBN' +
        os.sep +
        filename[1] +
        os.sep +
        filename[1] +
        '_newnss.wav',
        wav_newnss,
        16000)
    librosa.output.write_wav(
        'lenthBN' +
        os.sep +
        filename[1] +
        os.sep +
        filename[1] +
        '_kaelman.wav',
        wav_kaelman,
        16000)
    librosa.output.write_wav(
        'lenthBN' +
        os.sep +
        filename[1] +
        os.sep +
        filename[1] +
        '_weina.wav',
        wav_weina,
        16000)
    librosa.output.write_wav(
        'lenthBN' +
        os.sep +
        filename[1] +
        os.sep +
        filename[1] +
        '_segan.wav',
        wav_segan,
        16000)
    librosa.output.write_wav(
        'lenthBN' +
        os.sep +
        filename[1] +
        os.sep +
        filename[1] +
        '_noisy.wav',
        wav_noisy,
        16000)

