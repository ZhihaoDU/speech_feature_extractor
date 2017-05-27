# coding = utf-8
import numpy as np
from read_sphere_wav import  read_sphere_wav
from feature_extractor import cochleagram_extractor
from matplotlib import  pyplot as plt


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
    wav_data, wav_header = read_sphere_wav(u"/media/neo/000C6F0F00042510/Doctor/dataset/TIMIT/train/dr1/fcjf0/sa1.wav")
    cochlea = cochleagram_extractor(wav_data, 320, 160, 64, 'hanning')
    gfcc = gfcc_extractor(cochlea, 64, 31)
    plt.figure()
    plt.subplot(211)
    plt.imshow(np.flipud(cochlea))
    plt.subplot(212)
    plt.imshow(gfcc[1:, :])
    plt.show()
