import numpy as np
from scipy import signal
from scipy import fftpack


def power_spectrum(data: np.ndarray, fs: int = 44100) -> np.ndarray:
    ''' calculate power spectrum

    zero padding at two times the length of data

    Args:
        data (np.ndarray): signal of sounds
        fs (int): sampling frequency

    Returns:
        power (np.ndarray): power spectrum
        freq (np.ndarray) : frequency

    '''
#     _n = len(data)
    _n = int(len(data)*2)
    power = fftpack.fft(data, n=_n)
    power = abs(power)[0:round(_n / 2)] ** 2 / _n
    power = 10 * np.log10(power)
    freq = np.arange(0, fs / 2, fs / _n)
    return power, freq


def _hamming(data):
    win_hamming = signal.hamming(len(data))
    return data * win_hamming


def stft(data, window=500, slide=50, fs=50):
    th = -30 # dB
    power = []
    freq = []
    for i in range(0, len(data)-window, slide):
        p, f = power_spectrum(_hamming(data[i:i+window]), fs)
        power.append(p)
        freq.append(f)
    time = np.linspace(0, round(len(data)/fs), len(power))
    power = np.array(power).T
    power = np.where(power>th, power, -100)
    return power, freq[0], time
