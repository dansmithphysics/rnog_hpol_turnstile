from scipy.signal import butter, lfilter, cheby1
import numpy as np

def get_efield(filename):
    data = np.genfromtxt(filename, dtype = "float", delimiter = ",", skip_header = 1)
    ts = np.array(data[:,0])
    efield = np.array(data[:,1])
    return ts, efield

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band') #butter
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def phase_shift(efield, phase):
    efield_fft = np.fft.rfft(efield)
    efield_fft_mag = np.abs(efield_fft)
    efield_fft_ang = np.unwrap(np.angle(efield_fft))
    efield_fft_ang += phase
    efield_fft = efield_fft_mag * (np.cos(efield_fft_ang) + 1j * np.sin(efield_fft_ang))
    efield = np.fft.irfft(efield_fft)
    return efield
