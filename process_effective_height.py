import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import glob
import scipy.interpolate
import scipy.integrate
from scipy.signal import butter, lfilter, cheby1
from scipy.spatial.transform import Rotation as R
import pickle 

def get_efield(filename):

    f = open(filename)
    ts = []
    efield = []
    for iLine, line in enumerate(f):
        if(iLine == 0):
            continue

        line_ = line.split(",")

        ts += [float(line_[0])]
        efield += [float(line_[1])]

    return np.array(ts), np.array(efield)

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

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return r, el, az

def zero_out_zero(h_fft):
    h_fft_ang = np.angle(h_fft)
    h_fft_mag = np.abs(h_fft)

    h_fft_ang[0] = 0.0
    h_fft_mag[0] = 0.0

    h_fft_ = h_fft_mag * (np.cos(h_fft_ang) + 1j * np.sin(h_fft_ang))
    return h_fft_

def open_csv(filename, nrows):
    f = open(filename)
    data = [[] for i in range(nrows)]
    for iLine, line in enumerate(f):
        if(iLine == 0):
            continue
        line_ = line.split(",")
        for i in range(nrows):
            data[i] += [float(line_[i])]
    return data

def phase_shift(efield, phase):
    efield = np.roll(efield, 0)
    #efield_fft = np.fft.rfft(efield)
    #efield_fft_mag = np.abs(efield_fft)
    #efield_fft_ang = np.unwrap(np.angle(efield_fft))
    #efield_fft_ang += phase
    #efield_fft = efield_fft_mag * (np.cos(efield_fft_ang) + 1j * np.sin(efield_fft_ang))
    #efield = np.fft.irfft(efield_fft)
    return efield

n = 1.74
c = 0.3
r = 2.6
ZL = 50.0
Z0 = 120.0 * np.pi

ts = [[], []]
V_measured = [[], []]
s11_freqs = [[], []]
s11_r = [[], []]
s11_i = [[], []]
s11 = [np.array([]), np.array([])]
V_measured_func = [[], []]

base_names = ["rnog_hpol_turnstile_model.xf/hpol_component_output", "rnog_hpol_turnstile_model.xf/hpol_component1_output"]
component_names = ["Component", "Component 1"]
for base_name, component_name in zip(base_names, component_names):
    print(base_name, component_name)

    index = 0
    if(base_name == "hpol_component1_output"):
        index = 1

    data = open_csv(base_name+"/"+component_name+"-Voltage (V).csv", 2)
    ts[index] = np.array(data[0])
    V_measured[index] = np.array(data[1])

    data = open_csv(base_name+"/"+component_name+"-S-ParametersImag.csv", 2)
    s11_freqs[index] = np.array(data[0])
    s11_i[index] = np.array(data[1])

    data = open_csv(base_name+"/"+component_name+"-S-ParametersReal.csv", 2)
    s11_freqs[index] = np.array(data[0])
    s11_r[index] = np.array(data[1])

    s11_freqs[index] = np.array(s11_freqs[index])
    s11[index] = np.array(s11_r[index]) + 1j * np.array(s11_i[index])

    V_measured_func[index] = scipy.interpolate.interp1d(ts[index], V_measured[index], kind='cubic', bounds_error=False, fill_value="extrapolate")

ts = ts[0]
V_measured = V_measured[0]
s11_freqs = s11_freqs[0]
s11_i = s11_i[0]
s11_r = s11_r[0]
s11 = s11[0]
V_measured_func = V_measured_func[0]

gain_circle = []
freq = 0.6

for azimuth_angle in np.arange(0, 100, 10):

    file_name_run1 = glob.glob(base_names[0]+"/Point Sensor 0 1 "+str(azimuth_angle)+"*X.csv")[0]
    file_name_run2 = glob.glob(base_names[1]+"/Point Sensor 0 1 "+str(azimuth_angle)+"*X.csv")[0]

    ts_run1, efield_x_run1 = get_efield(file_name_run1)
    ts_run1, efield_y_run1 = get_efield(file_name_run1.replace("X", "Y"))
    ts_run1, efield_z_run1 = get_efield(file_name_run1.replace("X", "Z"))

    ts_run2, efield_x_run2 = get_efield(file_name_run2)
    ts_run2, efield_y_run2 = get_efield(file_name_run2.replace("X", "Y"))
    ts_run2, efield_z_run2 = get_efield(file_name_run2.replace("X", "Z"))

    #plt.plot(ts_run1, efield_x_run1 * np.conj(efield_x_run1) + efield_y_run1 * np.conj(efield_y_run1), label="r", color="blue")
    #plt.plot(ts_run2, efield_x_run2 * np.conj(efield_x_run2) + efield_y_run2 * np.conj(efield_y_run2), label="r", color="red")

    # Phase shift
    efield_x_run2 = phase_shift(efield_x_run2, np.pi / 2.0)
    efield_y_run2 = phase_shift(efield_y_run2, np.pi / 2.0)
    efield_z_run2 = phase_shift(efield_z_run2, np.pi / 2.0)

    #plt.plot(ts_run2, efield_x_run2 * np.conj(efield_x_run2) + efield_y_run2 * np.conj(efield_y_run2), label="r", color="red", linestyle="--")

    # Time to combine
    f_efield_x_run1 = scipy.interpolate.interp1d(ts_run1, efield_x_run1, kind='cubic', bounds_error=False, fill_value="extrapolate")
    f_efield_y_run1 = scipy.interpolate.interp1d(ts_run1, efield_y_run1, kind='cubic', bounds_error=False, fill_value="extrapolate")
    f_efield_z_run1 = scipy.interpolate.interp1d(ts_run1, efield_z_run1, kind='cubic', bounds_error=False, fill_value="extrapolate")

    f_efield_x_run2 = scipy.interpolate.interp1d(ts_run2, efield_x_run2, kind='cubic', bounds_error=False, fill_value="extrapolate")
    f_efield_y_run2 = scipy.interpolate.interp1d(ts_run2, efield_y_run2, kind='cubic', bounds_error=False, fill_value="extrapolate")
    f_efield_z_run2 = scipy.interpolate.interp1d(ts_run2, efield_z_run2, kind='cubic', bounds_error=False, fill_value="extrapolate")

    efield_x_run1 = f_efield_x_run1(ts_run2)
    efield_y_run1 = f_efield_y_run1(ts_run2)
    efield_z_run1 = f_efield_z_run1(ts_run2)

    efield_x_run2 = f_efield_x_run2(ts_run2)
    efield_y_run2 = f_efield_y_run2(ts_run2)
    efield_z_run2 = f_efield_z_run2(ts_run2)

    ts = ts_run2
    efield_x = efield_x_run1 + efield_x_run2
    efield_y = efield_y_run1 + efield_y_run2
    efield_z = efield_z_run1 + efield_z_run2

    file_name = file_name_run1

    #plt.plot(ts, efield_x * np.conj(efield_x) + efield_y * np.conj(efield_y), label="r", color="purple")

    V_input = V_measured_func(ts)
    V_straight_fft = np.fft.rfft(V_input)
        
    #efield_x[ts > 45.0] = 0
    #efield_y[ts > 45.0] = 0
    #efield_z[ts > 45.0] = 0

    #efield_z = np.sqrt(np.square(efield_x) + np.square(efield_y) + np.sqrt(efield_z))

    # Convert cartesian to spherical
    ang = np.pi / 2.0 - np.deg2rad(float(file_name.split(" ")[4].split("-")[0]))

    x = r * np.cos(ang) * np.sin(np.pi / 2.0)
    y = r * np.sin(ang) * np.sin(np.pi / 2.0)
    z = r * np.cos(np.pi / 2.0)

    rho, el, az = cart2sph(x, y, z)
    the = np.pi/2.0 - el
    phi = az
    
    r_matrix = np.array([[np.sin(the) * np.cos(phi), np.cos(the) * np.cos(phi), np.cos(the)],
                         [np.cos(the) * np.cos(phi), np.cos(the) * np.sin(phi), -np.sin(the)],
                         [-np.sin(phi), np.cos(phi), 0.0]])
    
    efield_the = np.zeros(len(efield_x))    
    efield_r = np.zeros(len(efield_x))
    efield_phi = np.zeros(len(efield_x))
    
    for iii in range(len(efield_r)):
        temp_ = r_matrix.dot([efield_x[iii], efield_y[iii], efield_z[iii]])
        efield_r[iii]   = temp_[0]
        efield_the[iii] = temp_[1]
        efield_phi[iii] = temp_[2]
        
    efield_the_fft = np.fft.rfft(efield_the)
    efield_phi_fft = np.fft.rfft(efield_phi)

    # At this point, I assume the r component doesn't exist. Only fair in true far field. At 5m, it is  ~20 dB down in power

    freqs = np.fft.rfftfreq(len(efield_x), ts[1] - ts[0])
    
    res11_func = scipy.interpolate.interp1d(s11_freqs, np.real(s11).astype(float), kind='cubic', bounds_error=False, fill_value=0.0)
    ims11_func = scipy.interpolate.interp1d(s11_freqs, np.imag(s11), kind='cubic', bounds_error=False, fill_value=0.0)
    s11_ = res11_func(freqs) + 1j * ims11_func(freqs)
    V_straight_fft /= (1.0 + s11_) 
        
    '''
    plt.plot(freqs, (1.0 + np.abs(s11_)) / (1.0 - np.abs(s11_)), color="purple")
    plt.title("In-Ice VSWR of 4\" VPol Dipole, 65 cm Tall, in 11.2\" Borehole")    
    plt.xlim(0.1, 1.0)
    plt.ylim(1.0, 10.0)
    plt.minorticks_on()
    plt.grid(which="major")
    plt.grid(which="minor", alpha=0.25)
    plt.xlabel("Freq. [GHz]")
    plt.xlabel("VSWR")
    plt.show()
    exit()
    '''
    
    w = np.array(2.0 * np.pi * freqs)
    w[0] = 1.0
        
    h_fft_the = (2.0 * np.pi * r * c) / (1j * w) * (efield_the_fft / V_straight_fft) * (ZL / Z0) 
    h_fft_phi = (2.0 * np.pi * r * c) / (1j * w) * (efield_phi_fft / V_straight_fft) * (ZL / Z0) 
        
    #h_fft_phi = zero_out_zero(h_fft_phi)    
    #h_fft_the = zero_out_zero(h_fft_the)    
    
    h_the = np.fft.irfft(h_fft_the)
    h_phi = np.fft.irfft(h_fft_phi)
        
    h_the = butter_bandpass_filter(h_the, 0.01, 1.0, 1.0 / (ts[1] - ts[0]), order=5)
    h_phi = butter_bandpass_filter(h_phi, 0.01, 1.0, 1.0 / (ts[1] - ts[0]), order=5)
        
    # Align zero time, approximately
    h_the = np.roll(h_the, -100)
    h_phi = np.roll(h_phi, -100)
    
    #plt.plot(h_the)
    #continue
        
    h_fft_phi = np.fft.rfft(h_phi)
    h_fft_the = np.fft.rfft(h_the)
                
    # Convert to gains
    gain_the = 4.0 * np.pi * np.power(freqs * np.abs(h_fft_the) * n / c, 2.0) * Z0 / ZL / n 
    gain_phi = 4.0 * np.pi * np.power(freqs * np.abs(h_fft_phi) * n / c, 2.0) * Z0 / ZL / n
    
    gain = 4.0 * np.pi * np.power(freqs * np.sqrt(np.square(np.abs(h_fft_the)) + np.square(np.abs(h_fft_phi)))* n / c, 2.0) * Z0 / ZL / n 

    gain_circle += [gain_phi[100]]
    
    plt.plot(10.0 * np.log10(gain_phi), color="blue", alpha=(azimuth_angle + 10.0) / 100.0, label=(azimuth_angle))
    plt.plot(10.0 * np.log10(gain_the), color="purple")
    #plt.plot(freqs, 10.0 * np.log10(gain), color="purple")


plt.plot([0,0], [0,0], label="Theta Component", color="purple")
plt.plot([0,0], [0,0], label="Phi Component", color="blue")
plt.title("In-Ice Realized Gain of Fat Dipole")
plt.xlabel("Freqs. [GHz]")
plt.ylabel("Realized Gain [dBi]")
plt.ylim(-10., 10.0)
plt.xlim(0.0, 1.00)    


plt.legend(loc="upper right")
plt.grid()

plt.figure()
plt.polar(np.deg2rad(np.arange(0, 100, 10)), 10.0 * np.log10(gain_circle))
ax=plt.gca()
ax.set_rlim(-5,6)
plt.show()
