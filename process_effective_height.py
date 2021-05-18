import glob
import pickle 
import argparse
import scipy.interpolate
from scipy.spatial.transform import Rotation as R
import processing_functions as pf
import matplotlib.pyplot as plt
import numpy as np

def main(r, n, component_name, base_names, output_file_name):
    c = scipy.constants.c / 1e9
    ZL = 50.0 # Impedance of coax / feed
    Z0 = 120.0 * np.pi # Impedance of free space

    # Load up the data that has to do with the feed
    data = np.genfromtxt(base_names[0]+"/"+component_name+"-Voltage (V).csv", dtype = "float", delimiter = ",", skip_header = 1)
    ts = np.array(data[:,0])
    V_measured  = np.array(data[:,1])
    V_measured_func = scipy.interpolate.interp1d(ts, V_measured, kind = "cubic", bounds_error = False, fill_value = "extrapolate")
    
    data = np.genfromtxt(base_names[0]+"/"+component_name+"-S-ParametersImag.csv", dtype = "float", delimiter = ",", skip_header = 1)
    s11_freqs = np.array(data[:,0])
    s11_i = np.array(data[:,1])
    
    data = np.genfromtxt(base_names[0]+"/"+component_name+"-S-ParametersReal.csv", dtype = "float", delimiter = ",", skip_header = 1)
    s11_freqs = np.array(data[:,0])
    s11_r = np.array(data[:,1])
    
    res11_func = scipy.interpolate.interp1d(s11_freqs, s11_r, kind = "cubic", bounds_error = False, fill_value = 0.0)
    ims11_func = scipy.interpolate.interp1d(s11_freqs, s11_i, kind = "cubic", bounds_error = False, fill_value = 0.0)

    azimuth_angles = np.arange(0, 100, 10)

    effective_height_results_the = [[] for i in range(len(azimuth_angles))]
    effective_height_results_phi = [[] for i in range(len(azimuth_angles))]

    # Scan over azimuth angles
    for i_azimuth_angle, azimuth_angle in enumerate(azimuth_angles):

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
        phase_shift = np.pi / 2.0
        efield_x_run2 = pf.phase_shift(efield_x_run2, phase_shift)
        efield_y_run2 = pf.phase_shift(efield_y_run2, phase_shift)
        efield_z_run2 = pf.phase_shift(efield_z_run2, phase_shift)

        # Signal delay as proxy for phase shift in the second antenna
        time_delay = 0.0
        efield_x_run2 = pf.time_delay(ts_run2, efield_x_run2, time_delay)
        efield_y_run2 = pf.time_delay(ts_run2, efield_y_run2, time_delay)
        efield_z_run2 = pf.time_delay(ts_run2, efield_z_run2, time_delay)

        # Combine the two electric fields, only works if they have the same run time
        ts = ts_run2
        efield_x = (efield_x_run1 + efield_x_run2) 
        efield_y = (efield_y_run1 + efield_y_run2) 
        efield_z = (efield_z_run1 + efield_z_run2) 
        
        # Convert cartesian efields into spherical components
        the = np.pi/2.0
        phi = np.pi/2.0 - np.deg2rad(azimuth_angle)
        
        r_matrix = np.array([[np.sin(the) * np.cos(phi), np.cos(the) * np.cos(phi), np.cos(the)],
                             [np.cos(the) * np.cos(phi), np.cos(the) * np.sin(phi), -np.sin(the)],
                             [-np.sin(phi), np.cos(phi), 0.0]])
        
        transform = r_matrix.dot([efield_x, efield_y, efield_z])

        # At this point, I assume the r component doesn"t exist. Only fair in true far field. At 5m, it is  ~20 dB down in power
        efield_the = transform[1,:]
        efield_phi = transform[2,:]
        
        freqs = np.fft.rfftfreq(len(efield_x), ts[1] - ts[0])
        efield_the_fft = np.fft.rfft(efield_the)
        efield_phi_fft = np.fft.rfft(efield_phi)            
        s11_ = res11_func(freqs) + 1j * ims11_func(freqs)

        # Input voltage at feed, corrected for matching
        V_input = V_measured_func(ts)
        V_straight_fft = np.fft.rfft(V_input)
        V_straight_fft /= (1.0 + s11_) 
        
        w = np.array(2.0 * np.pi * freqs)
        w[0] = 1e-20 # get rid of divide by zero issues
        
        h_fft_the = (2.0 * np.pi * r * c) / (1j * w) * (efield_the_fft / V_straight_fft) * (ZL / Z0) 
        h_fft_phi = (2.0 * np.pi * r * c) / (1j * w) * (efield_phi_fft / V_straight_fft) * (ZL / Z0) 
            
        h_the = np.fft.irfft(h_fft_the)
        h_phi = np.fft.irfft(h_fft_phi)

        # Bandpass out aphysical freqs
        h_the = pf.butter_bandpass_filter(h_the, 0.01, 1.0, 1.0 / (ts[1] - ts[0]), order=5)
        h_phi = pf.butter_bandpass_filter(h_phi, 0.01, 1.0, 1.0 / (ts[1] - ts[0]), order=5)
        
        # Align zero time, approximately. Sadly, I never automated this.
        h_the = np.roll(h_the, -100)
        h_phi = np.roll(h_phi, -100)
    
        h_fft_the = np.fft.rfft(h_the)
        h_fft_phi = np.fft.rfft(h_phi)

        effective_height_results_the[i_azimuth_angle] = h_the
        effective_height_results_phi[i_azimuth_angle] = h_phi
                
        # Convert to gains
        gain = 4.0 * np.pi * np.power(freqs * np.sqrt(np.square(np.abs(h_fft_the)) + np.square(np.abs(h_fft_phi)))* n / c, 2.0) * Z0 / ZL / n
        gain_the = 4.0 * np.pi * np.power(freqs * np.abs(h_fft_the) * n / c, 2.0) * Z0 / ZL / n 
        gain_phi = 4.0 * np.pi * np.power(freqs * np.abs(h_fft_phi) * n / c, 2.0) * Z0 / ZL / n
        
        # Get rid of log of zero issues
        gain_the[0] = 1e-20
        gain_phi[0] = 1e-20

    effective_height_results_the = np.array(effective_height_results_the)
    effective_height_results_phi = np.array(effective_height_results_phi)

    np.savez(output_file_name, ts = ts, h_the = effective_height_results_the, h_phi = effective_height_results_phi)

def plot_gain(file_name, n):
    c = scipy.constants.c / 1e9
    ZL = 50.0 # Impedance of coax / feed
    Z0 = 120.0 * np.pi # Impedance of free space

    data = np.load(file_name)

    ts = data["ts"]
    h_the = data["h_the"]
    h_phi = data["h_phi"]

    freqs = np.fft.rfftfreq(len(h_phi[0]), ts[1] - ts[0])

    plt.figure()
    for i_azimuth_angle, azimuth_angle in enumerate(np.arange(0, 100, 10)):

        h_fft_the = np.fft.rfft(h_the[i_azimuth_angle])
        h_fft_phi = np.fft.rfft(h_phi[i_azimuth_angle])
        
        # Convert to gains
        gain = 4.0 * np.pi * np.power(freqs * np.sqrt(np.square(np.abs(h_fft_the)) + np.square(np.abs(h_fft_phi)))* n / c, 2.0) * Z0 / ZL / n
        gain_the = 4.0 * np.pi * np.power(freqs * np.abs(h_fft_the) * n / c, 2.0) * Z0 / ZL / n 
        gain_phi = 4.0 * np.pi * np.power(freqs * np.abs(h_fft_phi) * n / c, 2.0) * Z0 / ZL / n
    
        # Get rid of log of zero issues
        gain[0] = 1e-20
        gain_the[0] = 1e-20
        gain_phi[0] = 1e-20

        plt.plot(freqs, 10.0 * np.log10(gain), color = "purple", alpha = (azimuth_angle + 10.0) / 100.0, label = azimuth_angle)

        #plt.plot(freqs, 10.0 * np.log10(gain_the), color="purple", alpha=(azimuth_angle + 10.0) / 100.0)
        #plt.plot(freqs, 10.0 * np.log10(gain_phi), color="blue", alpha=(azimuth_angle + 10.0) / 100.0, label=(azimuth_angle))

    plt.title("In-Ice Realized Gain at Boresight / $90^\circ$ Zenith of HPol Prototype: \n Two Fat Dipoles on Sides, Displaced by 25 cm and by $90^\circ$")
    plt.legend(loc = 'lower right', title = "Azimuth Angles")
    plt.xlabel("Freqs. [GHz]")
    plt.ylabel("Realized Gain [dBi]")
    plt.ylim(-5., 5.0)
    plt.xlim(0.0, 1.00)
    plt.grid()

def plot_gain_polar(file_name, n, freqs_oi):
    c = scipy.constants.c / 1e9
    ZL = 50.0 # Impedance of coax / feed
    Z0 = 120.0 * np.pi # Impedance of free space

    data = np.load(file_name)

    ts = data["ts"]
    h_the = data["h_the"]
    h_phi = data["h_phi"]

    freqs = np.fft.rfftfreq(len(h_phi[0]), ts[1] - ts[0])

    # Plot per freqs_oi
    gains_oi_the = [[] for i in range(len(freqs_oi))]
    gains_oi_phi = [[] for i in range(len(freqs_oi))]

    azimuth_angles = np.arange(0, 100, 10)

    for i_azimuth_angle, azimuth_angle in enumerate(azimuth_angles):

        h_fft_the = np.fft.rfft(h_the[i_azimuth_angle])
        h_fft_phi = np.fft.rfft(h_phi[i_azimuth_angle])
        
        # Convert to gains
        gain = 4.0 * np.pi * np.power(freqs * np.sqrt(np.square(np.abs(h_fft_the)) + np.square(np.abs(h_fft_phi))) * n / c, 2.0) * Z0 / ZL / n
        gain_the = 4.0 * np.pi * np.power(freqs * np.abs(h_fft_the) * n / c, 2.0) * Z0 / ZL / n 
        gain_phi = 4.0 * np.pi * np.power(freqs * np.abs(h_fft_phi) * n / c, 2.0) * Z0 / ZL / n
    
        # Get rid of log of zero issues
        gain_the[0] = 1e-20
        gain_phi[0] = 1e-20

        f_gain_the = scipy.interpolate.interp1d(freqs, gain_the, kind = "cubic", bounds_error = False, fill_value = "extrapolate")
        f_gain_phi = scipy.interpolate.interp1d(freqs, gain_phi, kind = "cubic", bounds_error = False, fill_value = "extrapolate")

        for i_freq_oi, freq_oi in enumerate(freqs_oi):
            gains_oi_the[i_freq_oi] += [f_gain_the(freq_oi)]
            gains_oi_phi[i_freq_oi] += [f_gain_phi(freq_oi)]

    fig, ax = plt.subplots(nrows = 1, ncols = len(freqs_oi), subplot_kw = {'projection': 'polar'}, figsize = (3 * len(freqs_oi), 3))
    fig.suptitle("HPol In-Ice Realized Gain, Azimuth Beam Pattern at Boresight / $90^\circ$ Zenith", fontsize=14)

    for irow, row in enumerate(ax):
        row.set_title(str(np.round(freqs_oi[irow] * 1000.0, 0))+" MHz")
        row.plot(np.deg2rad(azimuth_angles), 10.0 * np.log10(gains_oi_phi[irow]), color = "purple")
        row.plot(np.deg2rad(azimuth_angles) + 1.0 * np.pi / 2.0, 10.0 * np.log10(np.flip(gains_oi_phi[irow])), color = "purple")
        row.plot(np.deg2rad(azimuth_angles) + 2.0 * np.pi / 2.0, 10.0 * np.log10(gains_oi_phi[irow]), color = "purple")
        row.plot(np.deg2rad(azimuth_angles) + 3.0 * np.pi / 2.0, 10.0 * np.log10(np.flip(gains_oi_phi[irow])), color = "purple")
        row.set_rmax(5.0)
        row.set_rmin(-5.0)        
        row.set_xticklabels(['', '$45^\circ$', '', '$135^\circ$', '', '$225^\circ$', '', '$315^\circ$'])

def plot_vswr(component_name, base_name):

    # Load up the data that has to do with the feed    
    data = np.genfromtxt(base_name+"/"+component_name+"-S-ParametersImag.csv", dtype = "float", delimiter = ",", skip_header = 1)
    s11_freqs = np.array(data[:,0])
    s11_i = np.array(data[:,1])
    
    data = np.genfromtxt(base_name+"/"+component_name+"-S-ParametersReal.csv", dtype = "float", delimiter = ",", skip_header = 1)
    s11_freqs = np.array(data[:,0])
    s11_r = np.array(data[:,1])

    s11 = s11_r + 1j * s11_i

    plt.figure()
    plt.plot(1000.0 * s11_freqs, (1.0 + np.abs(s11)) / (1.0 - np.abs(s11)), color="purple")
    plt.title("In-Ice VSWR of one of two Horizontal VPols")
    plt.xlim(0.1, 1000.0)
    plt.ylim(1.0, 10.0)
    plt.minorticks_on()
    plt.grid(which="major")
    plt.grid(which="minor", alpha=0.25)
    plt.xlabel("Freq. [MHz]")
    plt.ylabel("VSWR")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run NuRadioMC simulation")
    parser.add_argument("--n", type=float,
                        help="Index of refraction of simulation",
                        default=1.74)
    parser.add_argument("--r", type=float,
                        help="Distance from antenna to near field sensors",
                        default=2.6)
    parser.add_argument("--componentname", type=str,
                        help="name of component file", 
                        default="Component")
    parser.add_argument("--basename1", type=str,
                        help="names of location of xfs two output directories", 
                        default="rnog_hpol_turnstile_model.xf/hpol_component_output")
    parser.add_argument("--basename2", type=str,
                        help="names of location of xfs two output directories", 
                        default="rnog_hpol_turnstile_model.xf/hpol_component1_output")
    parser.add_argument("--outputfilename", type=str,
                        help="name of effective height output",
                        default="hpol_processed_effective_height")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    main(r = args.r, 
         n = args.n,
         component_name = args.componentname,
         base_names = [args.basename1, args.basename2],
         output_file_name = args.outputfilename
    )

    plot_gain(args.outputfilename+".npz", args.n)
    plt.savefig("hpol_realized_gain.png")

    plot_vswr(args.componentname, args.basename1)

    plot_gain_polar(args.outputfilename+".npz", args.n, np.linspace(0.2, 0.5, 5))

    plt.show()
