# coding = utf-8
import numpy as np
from scipy.signal import lfilter, lfilter_zi, lfiltic
from scikits.talkbox.linpred.levinson_lpc import lpc


def hz2mel(f):
    return 2595. * np.log10(1. + f / 700.)


def mel2hz(z):
    return 700. * (np.power(10., z / 2595.) - 1.)


def get_window(win_len, win_type):
    if win_type == 'hanning':
        win_len += 2
        window = np.hanning(win_len)
        window = window[1: -1]
    elif win_type == 'hamming':
        win_len += 2
        window = np.hamming(win_len)
        window = window[1: -1]
    elif win_type == 'triangle':
        window = 1. - (np.abs(win_len + 1. - 2.*np.arange(0., win_len+2., 1.)) / (win_len+1.))
        window = window[1: -1]
    else:
        window = np.ones(win_len)
    return window


def get_fft_mel_mat(nfft, sr=8000, nfilts=None, width=1.0, minfrq=20, maxfrq=None, constamp=0):
    if nfilts is None:
        nfilts = nfft
    if maxfrq is None:
        maxfrq = sr // 2
    wts = np.zeros((nfilts, nfft//2+1))
    fftfrqs = np.arange(0, nfft//2+1) / (1. * nfft) * (sr)
    minmel = hz2mel(minfrq)
    maxmel = hz2mel(maxfrq)
    binfrqs = mel2hz(minmel + np.arange(0, nfilts+2) / (nfilts+1.) * (maxmel - minmel))
    # binbin = np.round(binfrqs / maxfrq * nfft)
    for i in range(nfilts):
        fs = binfrqs[[i+0, i+1, i+2]]
        fs = fs[1] + width * (fs - fs[1])
        loslope = (fftfrqs - fs[0]) / (fs[1] - fs[0])
        hislope = (fs[2] - fftfrqs) / (fs[2] - fs[1])
        wts[i, :] = np.maximum(0, np.minimum(loslope, hislope))
    return wts


def mfcc_extractor(xx, sr, win_len, shift_len, mel_channel, dct_channel, win_type, include_delta):

    my_melbank = get_fft_mel_mat(win_len, sr, mel_channel)

    pre_emphasis_weight = 0.9375

    # x = xx * (1-pre_emphasis_weight)
    x = np.append(xx[0], xx[1:] - pre_emphasis_weight * xx[:-1])
    dctcoef = np.zeros((dct_channel, mel_channel), dtype=np.float32)
    for i in range(dct_channel):
        n = np.linspace(0, mel_channel-1, mel_channel)
        dctcoef[i, :] = np.cos((2 * n + 1) * i * np.pi / (2 * mel_channel))

    w = 1 + 6 * np.sin(np.pi * np.linspace(0, dct_channel-1, dct_channel) / (dct_channel-1))
    w /= w.max()
    w = np.reshape(w, newshape=(dct_channel, 1))

    samples = x.shape[0]
    frames = 1 + (samples - win_len) // shift_len
    stft = np.zeros((win_len, frames), dtype=np.complex64)
    spectrum = np.zeros((win_len // 2 + 1, frames), dtype=np.float32)

    mfcc = np.zeros((dct_channel, frames), dtype=np.float32)

    window = get_window(win_len, win_type)

    for i in range(frames):
        one_frame = x[i * shift_len: i * shift_len + win_len]
        windowed_frame = np.multiply(one_frame, window)
        stft[:, i] = np.fft.fft(windowed_frame, win_len)
        spectrum[:, i] = np.power(np.abs(stft[0:win_len // 2 + 1, i]), 2)

    c1 = np.matmul(my_melbank, spectrum)
    c1 = np.where(c1 == 0.0, np.finfo(float).eps, c1)
    mfcc[:dct_channel, :] = np.multiply(np.matmul(dctcoef, np.log(c1)), np.repeat(w, frames, 1))

    if include_delta:
        dtm = np.zeros((dct_channel, frames), dtype=np.float32)
        ddtm = np.zeros((dct_channel, frames), dtype=np.float32)
        for i in range(2, frames-2):
            dtm[:, i] = 2 * mfcc[:, i+2] + mfcc[:, i+1] - mfcc[:, i-1] - 2 * mfcc[:, i-2]
        dtm /= 3.0
        for i in range(2, frames-2):
            ddtm[:, i] = 2 * dtm[:, i+2] + dtm[:, i+1] - dtm[:, i-1] - 2 * dtm[:, i-2]
        ddtm /= 3.0
        mfcc = np.row_stack((mfcc[:, 4:frames-4], dtm[:, 4:frames-4], ddtm[:, 4:frames-4]))

    return mfcc


def log_power_spectrum_extractor(x, win_len, shift_len, win_type, is_log=False):
    samples = x.shape[0]
    frames = 1 + (samples - win_len) // shift_len
    stft = np.zeros((win_len, frames), dtype=np.complex64)
    spect = np.zeros((win_len // 2 + 1, frames), dtype=np.float64)

    window = get_window(win_len, win_type)

    for i in range(frames):
        one_frame = x[i*shift_len: i*shift_len+win_len]
        windowed_frame = np.multiply(one_frame, window)
        stft[:, i] = np.fft.fft(windowed_frame, win_len)
        if is_log:
            spect[:, i] = np.log(np.power(np.abs(stft[0: win_len//2+1, i]), 2.))
        else:
            spect[:, i] = np.power(np.abs(stft[0: win_len//2+1, i]), 2.)

    return spect


def stft_extractor(x, win_len, shift_len, win_type, n_fft=None):
    if n_fft is None:
        n_fft = win_len
    samples = x.shape[0]
    frames = 1 + (samples - win_len) // shift_len
    stft = np.zeros((n_fft, frames), dtype=np.complex64)
    spect = np.zeros((n_fft // 2 + 1, frames), dtype=np.complex64)

    window = get_window(win_len, win_type)

    for i in range(frames):
        one_frame = x[i*shift_len: i*shift_len+win_len]
        windowed_frame = np.multiply(one_frame, window)
        stft[:, i] = np.fft.fft(windowed_frame, n_fft)
        spect[:, i] = stft[: n_fft//2+1, i]

    return spect


def erb_space(low_freq=50, high_freq=8000, n=64):
    ear_q = 9.26449
    min_bw = 24.7

    cf_array = -(ear_q * min_bw) + np.exp(np.linspace(1,n,n) * (-np.log(high_freq + ear_q * min_bw) + np.log(low_freq + ear_q * min_bw)) / n) \
                * (high_freq + ear_q * min_bw)
    return cf_array


def make_erb_filters(sr, num_channels, low_freq):
    t = 1. / sr
    cf = erb_space(low_freq, sr // 2, num_channels)

    ear_q = 9.26449
    min_bw = 24.7
    order = 4

    erb = np.power(np.power(cf/ear_q, order) + (min_bw ** order), 1. / order)
    b = 1.019 * 2 * np.pi * erb

    a0 = t
    a2 = 0
    b0 = 1
    b1 = -2 * np.cos(2 * cf * np.pi * t) / np.exp(b*t)
    b2 = np.exp(-2 * b * t)

    a11 = -(2 * t * np.cos(2*cf*np.pi*t) / np.exp(b*t) + 2 * np.sqrt(3+2**1.5) * t * np.sin(2*cf*np.pi*t) / np.exp(b*t))/2
    a12 = -(2 * t * np.cos(2*cf*np.pi*t) / np.exp(b*t) - 2 * np.sqrt(3+2**1.5) * t * np.sin(2*cf*np.pi*t) / np.exp(b*t))/2
    a13 = -(2 * t * np.cos(2*cf*np.pi*t) / np.exp(b*t) + 2 * np.sqrt(3-2**1.5) * t * np.sin(2*cf*np.pi*t) / np.exp(b*t))/2
    a14 = -(2 * t * np.cos(2*cf*np.pi*t) / np.exp(b*t) - 2 * np.sqrt(3-2**1.5) * t * np.sin(2*cf*np.pi*t) / np.exp(b*t))/2

    p1 = (-2*np.exp(4j*cf*np.pi*t)*t + 2*np.exp(-(b*t) + 2j*cf*np.pi*t) * t *
         (np.cos(2*cf*np.pi*t) - np.sqrt(3 - 2**(3/2))* np.sin(2*cf*np.pi*t)))
    p2 = (-2*np.exp(4j*cf*np.pi*t)*t + 2*np.exp(-(b*t) + 2j*cf*np.pi*t) * t *
         (np.cos(2*cf*np.pi*t) + np.sqrt(3 - 2**(3/2))* np.sin(2*cf*np.pi*t)))
    p3 = (-2*np.exp(4j*cf*np.pi*t)*t + 2*np.exp(-(b*t) + 2j*cf*np.pi*t) * t *
         (np.cos(2*cf*np.pi*t) - np.sqrt(3 + 2**(3/2))* np.sin(2*cf*np.pi*t)))
    p4 = (-2*np.exp(4j*cf*np.pi*t)*t + 2*np.exp(-(b*t) + 2j*cf*np.pi*t) * t *
         (np.cos(2*cf*np.pi*t) + np.sqrt(3 + 2**(3/2))* np.sin(2*cf*np.pi*t)))
    p5 = np.power(-2 / np.exp(2*b*t) - 2 * np.exp(4j*cf*np.pi*t) + 2 * (1 + np.exp(4j*cf*np.pi*t)) / np.exp(b*t), 4)
    gain = np.abs(p1 * p2 * p3 * p4 / p5)

    allfilts = np.ones((np.size(cf, 0), 1), dtype=np.float32)
    fcoefs = np.column_stack((a0*allfilts, a11, a12, a13, a14, a2*allfilts, b0*allfilts, b1, b2, gain))
    return fcoefs, cf


def erb_frilter_bank(x, fcoefs):
    a0 = fcoefs[:, 0]
    a11 = fcoefs[:, 1]
    a12 = fcoefs[:, 2]
    a13 = fcoefs[:, 3]
    a14 = fcoefs[:, 4]
    a2 = fcoefs[:, 5]
    b0 = fcoefs[:, 6]
    b1 = fcoefs[:, 7]
    b2 = fcoefs[:, 8]
    gain = fcoefs[:, 9]

    output = np.zeros((np.size(gain, 0), np.size(x, 0)))

    for chan in range(np.size(gain, 0)):
        y1 = lfilter(np.array([a0[chan] / gain[chan], a11[chan] / gain[chan], a2[chan] / gain[chan]]),
                     np.array([b0[chan], b1[chan], b2[chan]]), x)
        y2 = lfilter(np.array([a0[chan], a12[chan], a2[chan]]),
                     np.array([b0[chan], b1[chan], b2[chan]]), y1)
        y3 = lfilter(np.array([a0[chan], a13[chan], a2[chan]]),
                     np.array([b0[chan], b1[chan], b2[chan]]), y2)
        y4 = lfilter(np.array([a0[chan], a14[chan], a2[chan]]),
                     np.array([b0[chan], b1[chan], b2[chan]]), y3)

        output[chan, :] = y4
    return output


def cochleagram_extractor_wdl(xx, sr, win_len, shift_len, channel_number, win_type):
    fcoefs, f = make_erb_filters(sr, channel_number, 50)
    fcoefs = np.flipud(fcoefs)
    xf = erb_frilter_bank(xx, fcoefs)

    window = get_window(win_len, win_type)
    window = window.reshape((1, win_len))

    xe = np.power(xf, 2.0)
    frames = 1 + ((np.size(xe, 1)-win_len) // shift_len)
    cochleagram = np.zeros((channel_number, frames))
    for i in range(frames):
        one_frame = np.multiply(xe[:, i*shift_len:i*shift_len+win_len], np.repeat(window, channel_number, 0))
        cochleagram[:, i] = np.sum(one_frame, 1)
    return cochleagram


def cochleagram_extractor(xx, sr, win_len, shift_len, channel_number, win_type):
    fcoefs, f = make_erb_filters(sr, channel_number, 50)
    fcoefs = np.flipud(fcoefs)
    xf = erb_frilter_bank(xx, fcoefs)

    window = get_window(win_len, win_type)
    window = window.reshape((1, win_len))

    xe = np.power(xf, 2.0)
    frames = 1 + ((np.size(xe, 1)-win_len) // shift_len)
    cochleagram = np.zeros((channel_number, frames))
    for i in range(frames):
        one_frame = np.multiply(xe[:, i*shift_len:i*shift_len+win_len], np.repeat(window, channel_number, 0))
        cochleagram[:, i] = np.sqrt(np.mean(one_frame, 1))

    cochleagram = np.where(cochleagram == 0.0, np.finfo(float).eps, cochleagram)
    cochleagram = np.power(cochleagram, 1./3)
    return cochleagram


def fft_to_cochleagram(sr, min_freq, max_freq, win_len, channel_number):
    max_len = win_len
    nfilts = channel_number
    nfft = win_len

    wts = np.zeros((nfilts, nfft // 2 + 1))
    ear_q = 9.26449
    min_bw = 24.7
    order = 1.
    cfreqs = -(ear_q * min_bw) + np.exp(np.arange(1, nfilts+1, 1) * (-np.log(max_freq+ear_q*min_bw) + np.log(min_freq + ear_q*min_bw)) / nfilts) * (max_freq + ear_q*min_bw)
    cfreqs = np.flipud(cfreqs)
    GTord = 4.
    ucirc = np.exp(2j * np.pi * np.arange(0, nfft//2+1, 1)/nfft)

    for i in range(nfilts):
        cf = cfreqs[i]
        erb = 1.0 * np.power((np.power(cf/ear_q, order) + min_bw ** order), 1.0/order)
        b = 1.019 * 2 * np.pi * erb
        r = np.exp(-b / sr)
        theta = 2 * np.pi * cf / sr
        pole = r * np.exp(1j * theta)

        t = 1. / sr

        a11 = -(2 * t * np.cos(2 * cf * np.pi * t) / np.exp(b * t) + 2 * np.sqrt(3 + 2 ** 1.5) * t * np.sin(
            2 * cf * np.pi * t) / np.exp(b * t)) / 2
        a12 = -(2 * t * np.cos(2 * cf * np.pi * t) / np.exp(b * t) - 2 * np.sqrt(3 + 2 ** 1.5) * t * np.sin(
            2 * cf * np.pi * t) / np.exp(b * t)) / 2
        a13 = -(2 * t * np.cos(2 * cf * np.pi * t) / np.exp(b * t) + 2 * np.sqrt(3 - 2 ** 1.5) * t * np.sin(
            2 * cf * np.pi * t) / np.exp(b * t)) / 2
        a14 = -(2 * t * np.cos(2 * cf * np.pi * t) / np.exp(b * t) - 2 * np.sqrt(3 - 2 ** 1.5) * t * np.sin(
            2 * cf * np.pi * t) / np.exp(b * t)) / 2

        zros = -1 * np.column_stack((a11, a12, a13, a14))/t
        p1 = (-2 * np.exp(4j * cf * np.pi * t) * t + 2 * np.exp(-(b * t) + 2j * cf * np.pi * t) * t *
              (np.cos(2 * cf * np.pi * t) - np.sqrt(3 - 2 ** (3 / 2)) * np.sin(2 * cf * np.pi * t)))
        p2 = (-2 * np.exp(4j * cf * np.pi * t) * t + 2 * np.exp(-(b * t) + 2j * cf * np.pi * t) * t *
              (np.cos(2 * cf * np.pi * t) + np.sqrt(3 - 2 ** (3 / 2)) * np.sin(2 * cf * np.pi * t)))
        p3 = (-2 * np.exp(4j * cf * np.pi * t) * t + 2 * np.exp(-(b * t) + 2j * cf * np.pi * t) * t *
              (np.cos(2 * cf * np.pi * t) - np.sqrt(3 + 2 ** (3 / 2)) * np.sin(2 * cf * np.pi * t)))
        p4 = (-2 * np.exp(4j * cf * np.pi * t) * t + 2 * np.exp(-(b * t) + 2j * cf * np.pi * t) * t *
              (np.cos(2 * cf * np.pi * t) + np.sqrt(3 + 2 ** (3 / 2)) * np.sin(2 * cf * np.pi * t)))
        p5 = np.power(
            -2 / np.exp(2 * b * t) - 2 * np.exp(4j * cf * np.pi * t) + 2 * (1 + np.exp(4j * cf * np.pi * t)) / np.exp(
                b * t), 4)
        gain = np.abs(p1 * p2 * p3 * p4 / p5)

        wts[i, :] = ((t ** 4) / gain) * np.abs(ucirc - zros[:, 0]) * np.abs(ucirc - zros[:, 1]) * \
                    np.abs(ucirc - zros[:, 2]) * np.abs(ucirc - zros[:, 3]) * \
                    np.power(np.abs((pole - ucirc) * (np.conj(pole) - ucirc)), -1*GTord)

    return wts


def freq2bark(f):
    return 7.*np.log(f/650.+np.sqrt(np.power(1.+(f/650.), 2.)))


def bark2freq(b):
    return 650.*np.sinh(b/7.)


def get_fft_bark_mat(sr, fft_len, barks, min_frq=20, max_frq=None):
    if max_frq is None:
        max_frq = sr // 2
    fft_frqs = np.arange(0, fft_len//2+1) / (1.*fft_len) * sr
    min_bark = freq2bark(min_frq)
    max_bark = freq2bark(max_frq)
    bark_bins = bark2freq(min_bark + np.arange(0, barks+2) / (barks + 1.) * (max_bark - min_bark))
    wts = np.zeros((barks, fft_len//2+1))
    for i in range(barks):
        fs = bark_bins[[i+0, i+1, i+2]]
        loslope = (fft_frqs - fs[0]) / (fs[1] - fs[0])
        hislope = (fs[2] - fft_frqs) / (fs[2] - fs[1])
        wts[i, :] = np.maximum(0, np.minimum(loslope, hislope))
    return wts


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


def calc_normalized_autocorrelation(x, win_len, shift_len, Tn):
    from numpy.linalg import norm
    frame_number = 1 + (len(x) - win_len) // shift_len
    A = np.zeros(shape=(win_len // Tn, frame_number))
    for i in range(frame_number):
        one_frame = x[i*shift_len: i*shift_len+win_len]
        for t in range(1, win_len // Tn-1):
            n = np.arange(t*Tn, win_len, Tn)
            A[t, i] = np.sum(one_frame[n]*one_frame[n - t*Tn]) / (norm(one_frame[n]) * norm(one_frame[n - t*Tn]))
    return A


def calc_average_instaneous_frequence(ac_matrix, win_duration_ms):
    frames = np.size(ac_matrix, 1)
    average_if = np.zeros(frames)
    for i in range(frames):
        zero_cross_times = np.sum(np.less(ac_matrix[:-2, i] * ac_matrix[1:-1, i], 0))
        average_if[i] = 1. / (win_duration_ms / zero_cross_times)
    return average_if


def ams_extractor(x, sr, win_len, shift_len, order=1, decimate_coef=1./4.):
    from scipy.signal import hilbert
    envelope = np.abs(hilbert(x))
    for i in range(order-1):
        envelope = np.abs(hilbert(envelope))
    envelope = envelope * decimate_coef
    frames = 1 + (len(envelope) - win_len) // shift_len
    hanning_window = np.hanning(win_len)
    ams_feature = np.zeros(shape=(15, frames))
    wts = cal_triangle_window(0, sr//2, win_len//2+1, 15, 15.6, 401)
    for i in range(frames):
        one_frame = envelope[i*shift_len:i*shift_len+win_len]
        one_frame = one_frame * hanning_window
        frame_fft = np.abs(np.fft.fft(one_frame, win_len))
        frame_fft = frame_fft[:win_len//2+1]
        ams_feature[:,i] = np.matmul(wts, frame_fft)
    return ams_feature


def unknown_feature_extractor(x, sr, win_len, shift_len, barks, inner_win, inner_shift, win_type, method_version):
    x_spectrum = stft_extractor(x, win_len, shift_len, win_type)
    coef = get_fft_bark_mat(sr, win_len, barks, 20, sr//2)
    bark_spect = np.matmul(coef, x_spectrum)
    ams = np.zeros((barks, inner_win//2+1, (bark_spect.shape[1] - inner_win)//inner_shift))
    for i in range(barks):
        channel_stft = stft_extractor(bark_spect[i, :], inner_win, inner_shift, 'hanning')
        if method_version == 'v1':
            ams[i, :, :] = 20 * np.log(np.abs(channel_stft[:inner_win//2+1, :(bark_spect.shape[1] - inner_win)//inner_shift]))
        elif method_version == 'v2':
            channel_amplitude = np.abs(channel_stft[:inner_win//2+1, :(bark_spect.shape[1] - inner_win)//inner_shift])
            channel_angle = np.angle(channel_stft[:inner_win//2+1, :(bark_spect.shape[1] - inner_win)//inner_shift])
            channel_angle = channel_angle - (np.floor(channel_angle / (2.*np.pi)) * (2.*np.pi))
            ams[i, :, :] = np.power(channel_amplitude, 1./3.) * channel_angle
        else:
            ams[i, :, :] = np.abs(channel_stft)
    return ams


def rasta_filt(x):
    number = np.arange(-2., 3., 1.)
    number = -1. * number / np.sum(number*number)
    denom = np.array([1., -0.94])
    zi = lfilter_zi(number, 1)
    zi = zi.reshape(1, len(zi))
    zi = np.repeat(zi, np.size(x, 0), 0)
    y, zf = lfilter(number, 1, x[:,0:4], axis=1, zi=zi)
    y, zf = lfilter(number, denom, x, axis=1, zi=zf)
    return y


def get_equal_loudness(nfpts, fmax, fbtype=None):
    if fbtype is None:
        fbtype = 'bark'
    if fbtype == 'bark':
        bancfhz = bark2freq(np.linspace(0, freq2bark(fmax), nfpts))
    fsq = bancfhz * bancfhz
    ftmp = fsq + 1.6e5
    eql = ((fsq/ftmp)**2) * ((fsq + 1.44e6)/(fsq + 9.61e6))
    eql = eql.reshape(np.size(eql), 1)
    return eql


def postaud(x, fmax, fbtype=None):
    if fbtype is None:
        fbtype = 'bark'
    nbands = x.shape[0]
    nframes = x.shape[1]
    nfpts = nbands
    if fbtype == 'bark':
        bancfhz = bark2freq(np.linspace(0, freq2bark(fmax), nfpts))
    fsq = bancfhz * bancfhz
    ftmp = fsq + 1.6e5
    eql = ((fsq/ftmp)**2) * ((fsq + 1.44e6)/(fsq + 9.61e6))
    eql = eql.reshape(np.size(eql), 1)
    z = np.repeat(eql, nframes, axis=1) * x
    z = z ** (1./3.)
    y = np.vstack((z[1, :], z[1:nbands-1, :], z[nbands-2, :]))
    return y


def do_lpc(spec, order, axis=0, error_normal=False):
    coeff, error, k = lpc(spec, order, axis=axis)
    if error_normal:
        error = np.reshape(error, (1, len(error)))
        error = np.repeat(error, order+1, axis=axis)
        return coeff / error
    else:
        return coeff[1:, :]


def get_dct_coeff(in_channel, out_channel):
    dct_coef = np.zeros((out_channel, in_channel), dtype=np.float32)
    for i in range(out_channel):
        n = np.linspace(0, in_channel - 1, in_channel)
        dct_coef[i, :] = np.cos((2 * n + 1) * i * np.pi / (2 * in_channel))
    return dct_coef

# I cannot understand it, maybe it works...
def lpc2cep(a, nout=None):
    nin = np.size(a, 0)
    ncol = np.size(a, 1)
    order = nin - 1
    if nout is None:
        nout = order + 1
    c = np.zeros((nout, ncol))
    c[0, :] = -1. * np.log(a[0, :])
    renormal_coef = np.reshape(a[0,:], (1, ncol))
    renormal_coef = np.repeat(renormal_coef, nin, axis=0)
    a = a / renormal_coef
    for n in range(1, nout):
        sumn = np.zeros(ncol)
        for m in range(1, n+1):
            sumn = sumn + (n-m) * a[m, :] * c[n-m, :]
        c[n, :] = -1. * (a[n, :] + 1. / n * sumn)
    return c


def rasta_plp_extractor(x, sr, win_len, shift_len, plp_order=0, do_rasta=True):
    spec = log_power_spectrum_extractor(x, win_len, shift_len, 'hanning', False)
    bark_filters = int(np.ceil(freq2bark(sr//2)))
    wts = get_fft_bark_mat(sr, win_len, bark_filters)
    bark_spec = np.matmul(wts, spec)
    if do_rasta:
        bark_spec = np.where(bark_spec == 0.0, np.finfo(float).eps, bark_spec)
        log_bark_spec = np.log(bark_spec)
        rasta_log_bark_spec = rasta_filt(log_bark_spec)
        bark_spec = np.exp(rasta_log_bark_spec)
    post_spec = postaud(bark_spec, sr/2.)
    # post_spec = bark_spec
    if plp_order > 0:
        lpcas = do_lpc(post_spec, plp_order)
    else:
        lpcas = post_spec
    return lpcas


def enframe_extractor(x, win_len, shift_len, win_type, delta_size=0):
    frame_num = 1 + (len(x)- win_len) // shift_len
    frames = np.zeros([win_len, frame_num], dtype=np.float32)
    window = get_window(win_len, win_type)
    for i in range(frame_num):
        frames[:, i] = x[i*shift_len: i*shift_len+win_len] * window
    if delta_size > 0:
        frames = frames[:, delta_size: -delta_size]
    return frames


def MfccGFAmsPlp_feature_extractor(xx, sr, win_len, win_shift, win_type, include_delta, arma_m=0):
    mfcc = mfcc_extractor(xx, sr, win_len, win_shift, 64, 31, win_type, False)
    cochleagram = cochleagram_extractor_wdl(xx, sr, win_len, win_shift, 64, win_type)
    cochleagram = np.power(cochleagram, 1./15.)
    ams = ams_extractor(xx, sr, win_len, win_shift)
    rasta_plp = rasta_plp_extractor(xx, sr, win_len, win_shift, plp_order=12, do_rasta=True)
    features = np.concatenate([mfcc, cochleagram, ams, rasta_plp], axis=0)
    if include_delta:
        delta_features = 2 * features[:, 4:] + features[:, 3:-1] - features[:, 1:-3] - 2 * features[:, 0:-4]
        delta_features = 1. / 3. * delta_features
        features = np.concatenate((features[:, 2:-2], delta_features), axis=0)
    if arma_m > 0:
        arma_feature = np.zeros_like(features)
        arma_feature[:, :arma_m] = features[:, :arma_m]
        for i in range(arma_m, features.shape[1]-arma_m):
            arma_feature[:, i] = features[:, i]
            for j in range(1, arma_m+1):
                arma_feature[:, i] += (arma_feature[:, i-j] + features[:, i+j])
            arma_feature[:, i] /= (2. * arma_m + 1)
        features = arma_feature[:, arma_m: -arma_m]
    return features
