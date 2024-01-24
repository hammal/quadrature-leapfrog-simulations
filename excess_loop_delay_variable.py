from dataclasses import dataclass
import pickle
from typing import Any, Dict, Iterable
import logging
import simset
import numpy as np
import cbadc
import matplotlib.pyplot as plt
import os
import subprocess
from calib_python import plot_impulse_response, bode_plot, plot_state_dist
import tikzplotlib
import sys

logger = logging.getLogger(__name__)
plt.rcParams["figure.figsize"] = [10.0, 7.5]
K_ref_half = 1 << 9

plt.style.use("ggplot")

simset.script_name = "excess_loop_delay_variable.py"
simset.concurrent_jobs = 48
simset.python_interpreter = sys.executable

## simset.data_folders.append(["path1", "path2", "path3"])

figures_folder = os.path.join(os.getcwd(), "figures/excess_loop_delay_variable")
os.makedirs(figures_folder, exist_ok=True)

tex_folder = os.path.join(os.getcwd(), "tex/excess_loop_delay_variable")
os.makedirs(tex_folder, exist_ok=True)

# Set simulation accuracy tolerances
atol = 1e-12
rtol = 1e-8


#######################################################################################
# A data class to accommodate the resulting data.
#######################################################################################
@dataclass
class ResultDataClass:
    """
    A dataclass to hold return arguments.
    """

    args: Dict
    res: Any
    time: Dict


def evaluate(
    u_hat: np.ndarray, fp: float, fs: float, BW_half: float, psd_size: int = 1 << 14
):
    res = {}
    for l in range(2):
        f, psd = cbadc.utilities.compute_power_spectral_density(
            u_hat[l, :],
            fs=fs,
            nperseg=psd_size,
        )
        signal_index = cbadc.utilities.find_sinusoidal(psd, 15)
        noise_index = np.ones(psd.size, dtype=bool)
        noise_index[signal_index] = False
        noise_index[f < (fp - BW_half)] = False
        noise_index[f > (fp + BW_half)] = False
        fom = cbadc.utilities.snr_spectrum_computation_extended(
            psd, signal_index, noise_index, fs=fs
        )
        est_SNR = cbadc.fom.snr_to_dB(fom["snr"])
        est_ENOB = cbadc.fom.snr_to_enob(est_SNR)
        res[f"u_hat_{l}"] = u_hat[l, :]
        res[f"psd_{l}"] = psd
        res[f"est_SNR_{l}"] = est_SNR
        res[f"est_ENOB_{l}"] = est_ENOB
    res["f"] = f
    res["fp"] = fp
    res["fs"] = fs
    res["t"] = np.arange(u_hat.size) / fs
    return res


def evaluate_baseband(u_hat: np.ndarray, fs: float, BW: float, psd_size: int = 1 << 14):
    f, psd = cbadc.utilities.compute_power_spectral_density(
        u_hat,
        fs=fs,
        nperseg=psd_size,
    )
    signal_index = cbadc.utilities.find_sinusoidal(psd, 15)
    # if not signal_index:
    #     signal_index = [0]
    noise_index = np.ones(psd.size, dtype=bool)
    noise_index[signal_index] = False
    noise_index[f < (BW * 1e-2)] = False
    noise_index[f > BW] = False
    fom = cbadc.utilities.snr_spectrum_computation_extended(
        psd, signal_index, noise_index, fs=fs
    )
    est_SNR = cbadc.fom.snr_to_dB(fom["snr"])
    est_ENOB = cbadc.fom.snr_to_enob(est_SNR)
    return {
        "u_hat": u_hat,
        "psd": psd,
        "f": f,
        "est_SNR": est_SNR,
        "est_ENOB": est_ENOB,
        "t": np.arange(u_hat.size) / fs,
    }


fs = 1 << 31

# N = [4]
# OSR = 8

N = [6]
OSR = 4

# N = [6]
# OSR = 8

number_of_freq_points = OSR

fp_fs = 1 / (4 * OSR) + np.arange(number_of_freq_points) / float(
    number_of_freq_points - 1
) * (0.5 - 1 / (2 * OSR))

f_BW = 0.5 / OSR

k0 = 0.1

delta_DC = np.linspace(0.0, 1.0 / (1.0 * fs), 100)


#######################################################################################
# The simulate function to be executed at each node.
#######################################################################################
@simset.arg("N", N)
@simset.arg("fs", [fs])
@simset.arg("OSR", [OSR])
@simset.arg("fp_fs", fp_fs)
@simset.arg("input_amplitude", [1e0])
@simset.arg("target_size", [1 << 14])
@simset.arg("K1", [1 << 8])
@simset.arg("K2", [1 << 8])
@simset.arg("phi", [0.0])
@simset.arg("delta_DC", delta_DC)
@simset.arg("kappa_0_scale", [k0])
@simset.arg("training_size", [1 << 14])  # 1<< 18
@simset.arg("test_size", [1 << 14])
@simset.arg("training_iterations", [1 << 26])  # 1 << 27
@simset.arg("warm_up", [1 << 13])
def simulate_function(
    N,
    fs,
    OSR,
    fp_fs,
    input_amplitude,
    target_size,
    K1,
    K2,
    phi,
    delta_DC,
    kappa_0_scale,
    training_size,
    test_size,
    training_iterations,
    warm_up,
):
    """
    The simulate function which will run on each computer node
    with an unique argument tuple. Note that the order of the
    @simset.arg(...) and simulate_function(arg,...) must match.
    """
    logger.info("Starting simulation.")

    kwargs = {
        "N": N,
        "fs": fs,
        "OSR": OSR,
        "fp_fs": fp_fs,
        "input_amplitude": input_amplitude,
        "target_size": target_size,
        "K1": K1,
        "K2": K2,
        "phi": phi,
        "delta_DC": delta_DC,
        "kappa_0_scale": kappa_0_scale,
        "training_size": training_size,
        "test_size": test_size,
        "training_iterations": training_iterations,
        "warm_up": warm_up,
    }

    logger.info(f"fp_fs: {fp_fs}")
    logger.info(f"delta_DC: {delta_DC}")
    BW = 0.5 * fs / OSR

    fp = fp_fs * fs
    rel_delta_DC = delta_DC * fs

    unique_name = f"t_DAC_{N}_OSR={OSR}_delta_DC={rel_delta_DC}_fp_fs_{fp_fs:0.2f}"

    L = 2

    # Setup frontend
    analog_frontend_baseband = cbadc.synthesis.get_leap_frog(OSR=OSR * 2, N=N, T=1 / fs)

    fp = fp_fs * fs
    T = analog_frontend_baseband.digital_control.clock.T
    M = 2 * N  # + 2

    # align fp
    period_samples = 4 * int(np.ceil(1.0 / (fp_fs)))
    size = target_size

    analog_frontend_bandpass = cbadc.synthesis.get_bandpass(
        analog_frontend=analog_frontend_baseband, fc=fp, phi=phi, delta_DC=delta_DC
    )

    # Set delay time in digital control loop
    analog_frontend_bandpass.digital_control.t_delay = delta_DC

    # input signal
    u_hat = np.zeros((L, size))

    frequency_baseband = 1.0 / analog_frontend_bandpass.digital_control.clock.T
    while frequency_baseband > BW / 3:
        frequency_baseband /= 2

    in_phase = cbadc.analog_signal.Sinusoidal(
        input_amplitude, frequency_baseband + fp, phase=np.pi / 2 * 0
    )
    quadrature = cbadc.analog_signal.Sinusoidal(
        input_amplitude, frequency_baseband + fp, phase=np.pi / 2.0
    )

    logger.info(f"frequency: {frequency_baseband + fp}")

    # Training data simulation
    training_simulator = cbadc.simulator.FullSimulator(
        analog_frontend_bandpass.analog_system,
        analog_frontend_bandpass.digital_control,
        [quadrature, in_phase],
        atol=atol,
        rtol=rtol,
    )

    control_signals = np.zeros((size, M))
    time_vector_training = np.arange(size) * T
    states_training = np.zeros((size, 2 * N))

    for _ in range(warm_up):
        next(training_simulator)

    for index in range(size):
        control_signals[index, :] = next(training_simulator)
        states_training[index, :] = training_simulator.state_vector()

    # Setup Filters
    eta2 = (
        np.linalg.norm(
            analog_frontend_baseband.analog_system.transfer_function_matrix(
                np.array([np.pi * BW])
            )
        )
        ** 2
    )

    digital_estimator = cbadc.digital_estimator.FIRFilter(
        analog_frontend_bandpass.analog_system,
        analog_frontend_bandpass.digital_control,
        eta2,
        K1,
        K2,
    )

    numpy_simulator = cbadc.simulator.NumpySimulator("", control_signals)
    digital_estimator(numpy_simulator)

    for index in range(size):
        u_hat[:, index] = next(digital_estimator)

    # Plot the PSD of the digital estimator output.
    f, psd = cbadc.utilities.compute_power_spectral_density(
        u_hat[0, K1 << 1 :],
        fs=1 / T,
    )
    signal_index = cbadc.utilities.find_sinusoidal(psd, 75)
    noise_index = np.ones(psd.size, dtype=bool)
    noise_index[signal_index] = False
    noise_index[f < (fp - BW / 2.0)] = False
    noise_index[f > (fp + BW / 2.0)] = False
    noise_index[f < 1e6] = False
    noise_index[f > (fs / 2 - 1e7)] = False
    fom = cbadc.utilities.snr_spectrum_computation_extended(
        psd, signal_index, noise_index, fs=1 / T
    )
    est_SNR = cbadc.fom.snr_to_dB(fom["snr"])
    est_ENOB = cbadc.fom.snr_to_enob(est_SNR)

    # Plot
    figure_filename = os.path.join(
        figures_folder,
        unique_name,
    )

    os.makedirs(figures_folder, exist_ok=True)

    h = np.zeros((2, M, 2 * K1))
    for index in range(2):
        for m in range(M):
            h[index, m, :] = digital_estimator.h[index, :, m]

    plot_impulse_response(h, f"{figure_filename}_imp.png")
    bode_plot(h, f"{figure_filename}_bode.png", linear=True)

    plt.figure()
    plt.title("Analog state vectors")
    for index in range(2 * N):
        plt.plot(
            states_training[:, index],
            label="$x_{" + f"{index + 1}" + "}(t)$",
        )
    plt.grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
    plt.xlabel("$t/T$")
    plt.legend()

    plt.savefig(f"{figure_filename}.png")
    plt.close()
    plot_state_dist(states_training.transpose(), f"{figure_filename}_dist.png")
    plt.close()

    plt.figure()
    for index in range(M):
        plt.plot(
            (1.1**index) * control_signals[:, index],
            label="$s_{" + f"{index}" + "}(t)$",
        )
    plt.grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
    plt.xlabel("$t/T$")
    plt.legend()

    plt.savefig(f"{figure_filename}_control.png")
    plt.close()

    plot_state_dist(
        control_signals.transpose(),
        f"{figure_filename}_control_dist.png",
    )

    psd_plot = plt.figure()
    psd_axis = psd_plot.add_subplot(111)
    psd_axis.set_title("Power spectral density")
    psd_axis.set_xlabel("Hz")
    psd_axis.set_ylabel("V^2 / Hz dB")

    psd_axis.plot(
        f,
        10 * np.log10(np.abs(psd)),
        label=f"Q,N={N},SNR={est_SNR:.1f},OSR={OSR:.1f},amp={input_amplitude:.1f}",
    )

    f_I, psd_I = cbadc.utilities.compute_power_spectral_density(
        u_hat[1, K1 << 1 :],
        fs=1 / T,
        nperseg=test_size - K1 * 2,
    )

    psd_axis.plot(f_I, 10 * np.log10(np.abs(psd_I)), label="I")

    psd_axis.plot(f, 10 * np.log10(noise_index * np.abs(psd)), label="noise index")

    psd_axis.legend(loc="lower left")
    psd_plot.savefig(f"{figure_filename}_psd.png")
    plt.close(psd_plot)

    time_plot = plt.figure()
    time_plot_axis = time_plot.add_subplot(111)
    time_plot_axis.set_title("Estimate")
    time_plot_axis.set_xlabel("t [samples]")
    time_plot_axis.set_ylabel("V")

    time_plot_axis.plot(u_hat[0, :])

    time_plot_axis.legend()
    time_plot.savefig(f"{figure_filename}_time.png")
    plt.close(time_plot)

    result = {
        "state": {
            "time_vector": time_vector_training,
            "states": states_training,
            "control_signals": control_signals,
        },
        "state_training": {
            "time_vector": time_vector_training,
            "states": states_training,
        },
        "u_hat": u_hat,
        "OSR": OSR,
        "psd": {
            "f": f,
            "psd": psd,
            "signal_index": signal_index,
            "fom": fom,
            "est_SNR": est_SNR,
            "est_ENOB": est_ENOB,
        },
        "analog_frontend": analog_frontend_bandpass,
    }

    # put the results in the ResultDataClass
    result_dataclass = ResultDataClass(args=kwargs, res=result, time={})

    # return the results dataclass and it will automatically be stored
    return result_dataclass


TIME_PLOTS = True
PSD_PLOTS = True

dpi = 1200

threshold = 5.0


def outlier_remover(f, snr):
    snr_old = snr[0]

    f_new = []
    snr_new = []
    for ff, snrsnr in zip(f, snr):
        if snrsnr - snr_old < -threshold:
            print(f"outlier: {ff}, {snrsnr}")
        elif snrsnr > 100:
            print(f"outlier: {ff}, {snrsnr}")
        else:
            snr_old = snrsnr
            f_new.append(ff)
            snr_new.append(snrsnr)

    return np.array(f_new), np.array(snr_new)


#######################################################################################
# The post processing function for data merging.
#######################################################################################
def post_processing_function(results: Iterable[ResultDataClass]):
    """
    The post processing list where the argument is a iterable containing
    all currently finished simulation processes.
    """

    dpi = 1200

    # Testing and Training Errors

    t_DAC = {}
    SNR = {}
    L2_norm = {}
    L_infty = {}

    for res in results:
        keyname = f"fp={res.args['fp_fs']:.2f},N={res.args['N']},OSR={res.args['OSR']}"
        if keyname not in t_DAC:
            t_DAC[keyname] = []
            SNR[keyname] = []
            L2_norm[keyname] = []
            L_infty[keyname] = []

        t_DAC[keyname].append(res.args["delta_DC"] * res.args["fs"])
        SNR[keyname].append(res.res["psd"]["est_SNR"])
        L2_norm[keyname].append(np.std(res.res["state"]["states"][:, -1]))
        L_infty[keyname].append(
            np.linalg.norm(res.res["state"]["states"][:, -1], ord=np.inf)
        )

    plt.figure()
    plt.xlabel("$t_{DAC} / T$")
    plt.ylabel("SNR [dB]")
    plt.grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
    snr_ax = plt.gca()

    for key in t_DAC:
        t_DAC_temp, SNR_temp, L2_norm_temp, L_infty_temp = zip(
            *sorted(zip(t_DAC[key], SNR[key], L2_norm[key], L_infty[key]))
        )
        snr_ax.plot(t_DAC_temp, SNR_temp, label=key)
    plt.legend()
    plt.savefig(os.path.join(figures_folder, "SNR.png"), dpi=dpi)
    tikzplotlib.save(os.path.join(tex_folder, "SNR.tex"))
    plt.close()

    plt.figure()
    plt.xlabel("$t_{DAC} / T$")
    plt.ylabel("$L_2$ norm")
    plt.grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
    l2_ax = plt.gca()
    for key in t_DAC:
        t_DAC_temp, SNR_temp, L2_norm_temp, L_infty_temp = zip(
            *sorted(zip(t_DAC[key], SNR[key], L2_norm[key], L_infty[key]))
        )
        l2_ax.semilogy(t_DAC_temp, L2_norm_temp, label=f"avg$|x_N|_2^2$,{key}")
        l2_ax.semilogy(t_DAC_temp, L_infty_temp, label=f"$|x_N|_\infty$,{key}")
    plt.legend()
    plt.savefig(os.path.join(figures_folder, "L2_and_L_infty_norm.png"), dpi=dpi)
    tikzplotlib.save(os.path.join(tex_folder, "L2_and_L_infty_norm.tex"))
    plt.close()


#######################################################################################
# Specify a save function, defaults to pickeling.
#######################################################################################
def save(result: ResultDataClass, filename: str) -> None:
    """
    The save function, turning the ResultDataClass into
    a file on disk.
    """
    with open(filename, "wb") as f:
        pickle.dump(result, f, protocol=-1)


#######################################################################################
# Specify a load function, defaults to pickeling.
#######################################################################################
def load(filename: str) -> ResultDataClass:
    """
    A loader function loading the files, stored by save the save function,
    into a ResultDataClass object instance.
    """
    with open(filename, "rb") as f:
        return pickle.load(f)


#######################################################################################
# The main function typically needs no changing.
#
# To see the different execution modes of this file type:
#
# python simset_setup.py -h
#
#######################################################################################
if __name__ == "__main__":
    simset.command_line.command_line_simulate_process(
        simulate_function=simulate_function,
        process_function=post_processing_function,
        save=save,
        load=load,
    )
