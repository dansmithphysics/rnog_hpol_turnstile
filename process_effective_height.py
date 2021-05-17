import glob
import pickle 
import scipy.interpolate
from scipy.spatial.transform import Rotation as R
import processing_functions as pf
import matplotlib.pyplot as plt
import numpy as np

n = 1.74 # From the simulation, deep ice
c = scipy.constants.c / 1e9
r = 2.6 # From the simulation

ZL = 50.0 # Impedance of coax / feed
Z0 = 120.0 * np.pi # Impedance of free space

base_names = ["rnog_hpol_turnstile_model.xf/hpol_component_output", "rnog_hpol_turnstile_model.xf/hpol_component1_output"]
component_names = ["Component", "Component 1"]

def main():

    # Load up the data that have to do with the feed
    data = np.genfromtxt(base_names[0]+"/"+component_names[0]+"-Voltage (V).csv", dtype='float', delimiter=",", skip_header=1)
    ts = np.array(data[:,0])
    V_measured  = np.array(data[:,1])
    V_measured_func = scipy.interpolate.interp1d(ts, V_measured, kind='cubic', bounds_error=False, fill_value="extrapolate")
    
    data = np.genfromtxt(base_names[0]+"/"+component_names[0]+"-S-ParametersImag.csv", dtype='float', delimiter=",", skip_header=1)
    s11_freqs = np.array(data[:,0])
    s11_i = np.array(data[:,1])
    
    data = np.genfromtxt(base_names[0]+"/"+component_names[0]+"-S-ParametersReal.csv", dtype='float', delimiter=",", skip_header=1)
    s11_freqs = np.array(data[:,0])
    s11_r = np.array(data[:,1])
    
    res11_func = scipy.interpolate.interp1d(s11_freqs, s11_r, kind='cubic', bounds_error=False, fill_value=0.0)
    ims11_func = scipy.interpolate.interp1d(s11_freqs, s11_i, kind='cubic', bounds_error=False, fill_value=0.0)

    # Scan over azimuth angles
    for azimuth_angle in np.arange(0, 100, 10):

        # Load up electric fields at this azimuth
        file_name_run1 = glob.glob(base_names[0]+"/Point Sensor 0 1 "+str(azimuth_angle)+"*X.csv")[0]
        file_name_run2 = glob.glob(base_names[1]+"/Point Sensor 0 1 "+str(azimuth_angle)+"*X.csv")[0]
    
        ts_run1, efield_x_run1 = pf.get_efield(file_name_run1)
        ts_run1, efield_y_run1 = pf.get_efield(file_name_run1.replace("X", "Y"))
        ts_run1, efield_z_run1 = pf.get_efield(file_name_run1.replace("X", "Z"))
        
        ts_run2, efield_x_run2 = pf.get_efield(file_name_run2)
        ts_run2, efield_y_run2 = pf.get_efield(file_name_run2.replace("X", "Y"))
        ts_run2, efield_z_run2 = pf.get_efield(file_name_run2.replace("X", "Z"))
        
        # Phase shift the second antenna before combining
        efield_x_run2 = pf.phase_shift(efield_x_run2, np.pi / 2.0)
        efield_y_run2 = pf.phase_shift(efield_y_run2, np.pi / 2.0)
        efield_z_run2 = pf.phase_shift(efield_z_run2, np.pi / 2.0)

        # Combine the two electric fields, complicated if they have different run times
        ts = ts_run2
        efield_x = efield_x_run1 + efield_x_run2
        efield_y = efield_y_run1 + efield_y_run2
        efield_z = efield_z_run1 + efield_z_run2
        
        file_name = file_name_run1
        
        V_input = V_measured_func(ts)
        V_straight_fft = np.fft.rfft(V_input)
        
        # Convert cartesian efields into spherical components
        the = np.pi/2.0
        phi = np.pi/2.0 - np.deg2rad(azimuth_angle)
        
        r_matrix = np.array([[np.sin(the) * np.cos(phi), np.cos(the) * np.cos(phi), np.cos(the)],
                             [np.cos(the) * np.cos(phi), np.cos(the) * np.sin(phi), -np.sin(the)],
                             [-np.sin(phi), np.cos(phi), 0.0]])
        
        transform = r_matrix.dot([efield_x, efield_y, efield_z])

        # At this point, I assume the r component doesn't exist. Only fair in true far field. At 5m, it is  ~20 dB down in power

        efield_the = transform[1,:]
        efield_phi = transform[2,:]
        
        efield_the_fft = np.fft.rfft(efield_the)
        efield_phi_fft = np.fft.rfft(efield_phi)
        
        freqs = np.fft.rfftfreq(len(efield_x), ts[1] - ts[0])
        
        s11_ = res11_func(freqs) + 1j * ims11_func(freqs)
        V_straight_fft /= (1.0 + s11_) 
        
        w = np.array(2.0 * np.pi * freqs)
        w[0] = 1.0
        
        h_fft_the = (2.0 * np.pi * r * c) / (1j * w) * (efield_the_fft / V_straight_fft) * (ZL / Z0) 
        h_fft_phi = (2.0 * np.pi * r * c) / (1j * w) * (efield_phi_fft / V_straight_fft) * (ZL / Z0) 
            
        h_the = np.fft.irfft(h_fft_the)
        h_phi = np.fft.irfft(h_fft_phi)
        
        h_the = pf.butter_bandpass_filter(h_the, 0.01, 1.0, 1.0 / (ts[1] - ts[0]), order=5)
        h_phi = pf.butter_bandpass_filter(h_phi, 0.01, 1.0, 1.0 / (ts[1] - ts[0]), order=5)
        
        # Align zero time, approximately
        h_the = np.roll(h_the, -100)
        h_phi = np.roll(h_phi, -100)
    
        h_fft_phi = np.fft.rfft(h_phi)
        h_fft_the = np.fft.rfft(h_the)
                
        # Convert to gains
        gain = 4.0 * np.pi * np.power(freqs * np.sqrt(np.square(np.abs(h_fft_the)) + np.square(np.abs(h_fft_phi)))* n / c, 2.0) * Z0 / ZL / n
        gain_the = 4.0 * np.pi * np.power(freqs * np.abs(h_fft_the) * n / c, 2.0) * Z0 / ZL / n 
        gain_phi = 4.0 * np.pi * np.power(freqs * np.abs(h_fft_phi) * n / c, 2.0) * Z0 / ZL / n
        
        plt.plot(freqs, 10.0 * np.log10(gain_phi), color="blue", alpha=(azimuth_angle + 10.0) / 100.0, label=(azimuth_angle))
        plt.plot(freqs, 10.0 * np.log10(gain_the), color="purple")

    plt.plot([0,0], [0,0], label="Theta Component", color="purple")
    plt.plot([0,0], [0,0], label="Phi Component", color="blue")
    plt.title("In-Ice Realized Gain of Fat Dipole")
    plt.xlabel("Freqs. [GHz]")
    plt.ylabel("Realized Gain [dBi]")
    plt.ylim(-10., 10.0)
    plt.xlim(0.0, 1.00)    
    plt.show()

if __name__ == "__main__":
    main()
