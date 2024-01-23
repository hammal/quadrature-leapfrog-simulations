import numpy as np
from scipy.signal import find_peaks
import cbadc
import matplotlib.pyplot as plt
import os



figures_dir = "figures/transfer_function"
os.makedirs(figures_dir, exist_ok=True)

csv_dir = "csv/transfer_function"
os.makedirs(csv_dir, exist_ok=True)

# system order
N = [8, 4, 2]
# OSR = [4, 5, 8, 15, 16]
OSR = [4, 4, 4, 4, 4]
# sampling frequency
fs = 1 << 31
# notch frequency
fp = np.array([0.15, 0.3, 0.45, 0.4]) * fs
phi = 0.0
# excess loop delay
delta_DC = 0.0

inf = 300

omega = np.linspace(0, 1.0, 1000) * np.pi * fs
tf_results = []
tf_results_header = []

for index, (n, osr, fc) in enumerate(zip(N, OSR, fp)):
    # synthesize baseband analog frontend
    analog_frontend_baseband = cbadc.synthesis.get_leap_frog(
        OSR=osr * 2, N=n, T=1.0 / fs
    )
    # convert into a quadrauter analog frontend
    analog_frontend_bandpass = cbadc.synthesis.get_bandpass(
        analog_frontend=analog_frontend_baseband, fc=fc, phi=phi, delta_DC=delta_DC
    )
    # compute the baseband transfer function
    tf = analog_frontend_bandpass.analog_system.transfer_function_matrix(
        omega, symbolic=False
    )
    # compute the quadrature transfer function
    tf_lowpass = analog_frontend_baseband.analog_system.transfer_function_matrix(
        omega, symbolic=False
    )

    tf_1 = 20 * np.log10(np.abs(tf[-1, 0, :]))
    tf_2 = 20 * np.log10(np.abs(tf_lowpass[-1, 0, :]))
    peaks_index, _ = find_peaks(tf_1)
    peaks_index_lowpass, _ = find_peaks(tf_2)

    tf_1[peaks_index] = inf
    tf_2[peaks_index_lowpass] = inf

    # Visualize the results
    plt.plot(
        omega / (2 * np.pi * fs),
        tf_1,
        label=f"N={n}, OSR={osr}, fp={fc/fs:.2f}",
    )
    plt.plot(
        omega / (2 * np.pi * fs),
        tf_2,
        label=f"N={n}, OSR={osr}, fp=0",
    )

    # store the results for csv file generation
    tf_results.append(omega / (2 * np.pi * fs))
    tf_results.append(tf_1)
    tf_results.append(omega / (2 * np.pi * fs))
    tf_results.append(tf_2)
    tf_results_header.append(f"f_{index}")
    tf_results_header.append(f"tf_{index}")
    tf_results_header.append(f"f_lp_{index}")
    tf_results_header.append(f"tf_lp_{index}")

# save figure
plt.legend()
plt.xlabel("Normalized frequency")
plt.ylabel("Magnitude (dB)")
plt.ylim([0, 100])
plt.grid(True)
plt.title("Transfer function")
plt.savefig(os.path.join(figures_dir, "transfer_function.png"))


# save the results to csv files
np.savetxt(
    os.path.join(csv_dir, "transfer_functions.csv"),
    np.array(tf_results).transpose(),
    delimiter=",",
    fmt="%.6e",
    header=f"{','.join(tf_results_header)}",
    comments="",
)
