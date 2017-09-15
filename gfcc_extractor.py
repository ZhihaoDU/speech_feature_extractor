# coding = utf-8
import numpy as np
from scipy.io import wavfile
from feature_extractor import cochleagram_extractor
from matplotlib import  pyplot as plt
from speech_utils import read_sphere_wav


def gfcc_extractor(cochleagram, gf_channel, cc_channels):
    dctcoef = np.zeros((cc_channels, gf_channel))
    for i in range(cc_channels):
        n = np.linspace(0, gf_channel-1, gf_channel)
        dctcoef[i, :] = np.cos((2 * n + 1) * i * np.pi / (2 * gf_channel))
    plt.figure()
    plt.imshow(dctcoef)
    plt.show()
    return np.matmul(dctcoef, cochleagram)


if __name__ == '__main__':
    wav_data, wav_header = read_sphere_wav(u"sa1.wav")
    sr = 16000
    cochlea = cochleagram_extractor(wav_data, sr, 320, 160, 64, 'hanning')
    gfcc = gfcc_extractor(cochlea, 64, 31)
    plt.figure()
    plt.subplot(211)
    plt.imshow(np.flipud(cochlea))
    plt.subplot(212)
    plt.imshow(np.flipud(gfcc))
    plt.show()
