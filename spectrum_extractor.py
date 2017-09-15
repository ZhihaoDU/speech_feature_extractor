# coding=utf-8
from matplotlib import pyplot
import numpy as np
from scipy.io import wavfile


def spectrum_extractor(x, win_len, shift_len, win_type, is_log):
    samples = x.shape[0]
    frames = (samples - win_len) // shift_len
    stft = np.zeros((win_len, frames), dtype=np.complex64)
    spectrum = np.zeros((win_len // 2 + 1, frames), dtype=np.float64)

    if win_type == 'hanning':
        window = np.hanning(win_len)
    elif win_type == 'hamming':
        window = np.hamming(win_len)
    elif win_type == 'triangle':
        window = (1 - (np.abs(win_len - 1 - 2 * np.arange(1, win_len + 1, 1)) / (win_len + 1)))
    else:
        window = np.ones(win_len)
    for i in range(frames):
        one_frame = x[i*shift_len: i*shift_len+win_len]
        windowed_frame = np.multiply(one_frame, window)
        stft[:, i] = np.fft.fft(windowed_frame, win_len)
        if is_log:
            spectrum[:, i] = np.log(np.abs(stft[0: win_len//2+1, i]))
        else:
            spectrum[:, i] = np.abs(stft[0: win_len // 2 + 1:, i])

    return spectrum


if __name__ == '__main__':
    sr, wav_data = wavfile.read(u"clean.wav")
    spect = spectrum_extractor(wav_data, 160, 80, 'hamming', True)
    pyplot.imshow(spect)
    pyplot.show()
