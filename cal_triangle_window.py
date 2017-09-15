import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


def cal_triangle_window(min_freq, max_freq, nfft, window_number, low_freq, high_freq):
    fft_freq_bins = np.linspace(min_freq, max_freq, nfft)
    center_freq = np.linspace(low_freq, high_freq, window_number+2)
    wts = np.zeros(shape=(window_number, nfft))
    for i in range(window_number):
        fs = center_freq[[i+0, i+1, i+2]]
        fs = fs[1] + 1.0 * (fs - fs[1])
        loslope = (fft_freq_bins - fs[0]) / (fs[1] - fs[0])
        hislope = (fs[2] - fft_freq_bins) / (fs[2] - fs[1])
        wts[i, :] = np.maximum(0, np.minimum(loslope, hislope))
    return wts


def ams_extractor(x, sr, win_len, shift_len, order):
    from scipy.signal import hilbert
    envelope = np.abs(hilbert(x))
    for i in range(order-1):
        envelope = np.abs(hilbert(envelope))
    envelope = envelope * 1./3.
    frames = (len(envelope) - win_len) // shift_len
    hanning_window = np.hanning(win_len)
    ams_feature = np.zeros(shape=(15, frames))
    wts = cal_triangle_window(0, sr//2, win_len, 15, 15.6, 400)
    for i in range(frames):
        one_frame = x[i*shift_len:i*shift_len+win_len]
        one_frame = one_frame * hanning_window
        frame_fft = np.abs(np.fft.fft(one_frame, win_len))
        ams_feature[:,i] = np.matmul(wts, frame_fft)
    return ams_feature


if __name__ == '__main__':
    sr, wav_data = wavfile.read("clean.wav")
    ams_feature = ams_extractor(wav_data, sr, int(sr*0.032), int(sr*0.016), 1)
    plt.imshow(ams_feature)
    plt.show()