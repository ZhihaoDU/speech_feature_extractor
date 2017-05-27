# coding = utf-8

import numpy as np
from feature_extractor import *
from speech_utils import *
from matplotlib import pyplot as plt


def ideal_binary_mask(noisy_speech, clean_speech, snr):
    noisy_spect = stft_extractor(noisy_speech, 320, 160, 'hanning')
    clean_spect = stft_extractor(clean_speech, 320, 160, 'hanning')
    spect_snr = np.power(np.abs(clean_spect), 2.0) / np.power(np.abs(noisy_spect - clean_spect), 2.0)
    ibm = np.where(spect_snr > 10**(0.1*snr), 1, 0)

    return ibm


def synthesis_speech(ns, mk, win_type, win_len, shift_len, syn_method='A&R'):
    samples = ns.shape[0]
    frames = (samples - win_len) // shift_len

    if win_type == 'hanning':
        window = np.hanning(win_len)
    elif win_type == 'hamming':
        window = np.hamming(win_len)
    elif win_type == 'rectangle':
        window = np.ones(win_len)
    to_ifft = np.zeros(win_len, dtype=np.complex64)
    clean_speech = np.zeros((frames-1)*shift_len+win_len, dtype=np.float32)
    window_sum = np.zeros((frames-1)*shift_len+win_len, dtype=np.float32)
    for i in range(frames):
        one_frame = ns[i * shift_len: i * shift_len + win_len]
        windowed_frame = np.multiply(one_frame, window)
        stft = np.fft.fft(windowed_frame, win_len)
        masked_abs = np.abs(stft[:win_len//2+1]) * mk[:, i]
        to_ifft[:win_len//2+1] = masked_abs * np.exp(1j * np.angle(stft[:win_len//2+1]))
        to_ifft[win_len//2+1:] = np.conj(to_ifft[win_len//2-1:0:-1])
        speech_seg = np.real(np.fft.ifft(to_ifft, 320))

        if syn_method == 'A&R' or syn_method == 'ALLEN & RABINER':
            clean_speech[i*shift_len:i*shift_len+win_len] += speech_seg
            window_sum[i*shift_len:i*shift_len+win_len] += window

        elif syn_method == 'G&L' or syn_method == 'GRIFFIN & LIM':
            speech_seg = np.multiply(speech_seg, window)
            clean_speech[i * shift_len:i * shift_len + win_len] += speech_seg
            window_sum[i * shift_len:i * shift_len + win_len] += np.power(window, 2.)
        # if i > 0:
        #     clean_speech[i*shift_len: (i-1)*shift_len+win_len] *= 0.5
    window_sum = np.where(window_sum < 1e-2, 1e-2, window_sum)
    return clean_speech / window_sum


if __name__ == '__main__':
    speech_data, wav_header = read_sphere_wav(
        u"/media/neo/000C6F0F00042510/Doctor/dataset/TIMIT/train/dr1/fcjf0/sa1.wav")
    fs, noise_data = wavfile.read('/media/neo/000C6F0F00042510/Doctor/dataset/DEMAND/PCAFETER/ch01.wav')
    noisy_speech = mix_by_db(speech_data, noise_data, 5, 'cut')
    # mask = ideal_binary_mask(noisy_speech, speech_data, -5)
    mask = ideal_ratio_mask(noisy_speech, speech_data)
    # plt.imshow(mask)
    # plt.show()
    speech = synthesis_speech(noisy_speech, mask, 'hanning', 320, 160, 'G&L')
    wavfile.write('masked_speech_irm_GL', 16000, speech.astype(np.int16))
