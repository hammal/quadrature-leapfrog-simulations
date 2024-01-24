from dataclasses import dataclass
import pickle
from typing import Any, Dict, Iterable
import logging
import simset
import numpy as np
import cbadc
import matplotlib.pyplot as plt
import os
from cbadc.circuit.opamp import OpAmpFrontend
from cbadc.circuit.ota import GmCFrontend
from cbadc.circuit.testbench import OTATestBench, OpAmpTestBench
from cbadc.circuit.simulator import NGSpiceSimulator
from calib_python import plot_impulse_response, bode_plot, plot_state_dist
import scipy.signal
import sys

logger = logging.getLogger(__name__)
plt.rcParams["figure.figsize"] = [10.0, 7.5]
K_ref_half = 1 << 9

simset.script_name = "opamp.py"
simset.concurrent_jobs = 46
simset.python_interpreter = sys.executable

## simset.data_folders.append(["path1", "path2", "path3"])

netlist_folder = os.path.join(os.getcwd(), "opamp", "netlist")
ngspice_folder = os.path.join(os.getcwd(), "opamp", "ngspice")
figures_folder = os.path.join(os.getcwd(), "figures/opamp")
csv_folder = os.path.join(os.getcwd(), "csv/opamp")

os.makedirs(netlist_folder, exist_ok=True)
os.makedirs(ngspice_folder, exist_ok=True)
os.makedirs(figures_folder, exist_ok=True)


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

rel_GBWP = np.geomspace(1, 1e3, 25)
number_of_freq_points = OSR

fp_fs = 1 / (4 * OSR) + np.arange(number_of_freq_points) / float(
    number_of_freq_points - 1
) * (0.5 - 1 / (2 * OSR))

fp_fs = [fp_fs[-2]]

f_BW = 0.5 / OSR

k0 = 0.1

DC_gain = OSR / np.pi * np.array([20, 30, 50, 100, 500, 1e4])


#######################################################################################
# The simulate function to be executed at each node.
#######################################################################################
@simset.arg("N", N)
@simset.arg("fs", [fs])
@simset.arg("OSR", [OSR])
@simset.arg("fp_fs", fp_fs)
@simset.arg("input_amplitude", [1.0 - k0])
@simset.arg("target_size", [1 << 15])
@simset.arg("phi", [0.0])
@simset.arg("delta_DC", [0.0])
@simset.arg("kappa_0_scale", [k0])
@simset.arg("training_size", [1 << 14])  # 1<< 18
@simset.arg("test_size", [1 << 14])
@simset.arg("rel_GBWP", rel_GBWP)
@simset.arg("DC_gain", DC_gain)
@simset.arg("warm_up", [1 << 13])
@simset.arg("epochs", [1 << 10])
@simset.arg("K", [1 << 6])
def simulate_function(
    N,
    fs,
    OSR,
    fp_fs,
    input_amplitude,
    target_size,
    phi,
    delta_DC,
    kappa_0_scale,
    training_size,
    test_size,
    rel_GBWP,
    DC_gain,
    warm_up,
    epochs,
    K,
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
        "phi": phi,
        "delta_DC": delta_DC,
        "kappa_0_scale": kappa_0_scale,
        "training_size": training_size,
        "test_size": test_size,
        "rel_GBWP": rel_GBWP,
        "DC_gain": DC_gain,
        "warm_up": warm_up,
        "epochs": epochs,
        "K": K,
    }

    logger.info(f"fp_fs: {fp_fs}")
    BW = 0.5 * fs / OSR

    fp = fp_fs * fs
    GBWP = (fp + BW / 2.0) * rel_GBWP  # * 2.0 * np.pi

    unique_name = f"opamp_N_{N}_OSR={OSR}_GBWP={rel_GBWP:0.1e}_DC_gain={DC_gain * np.pi / OSR:0.1e}_fp_fs_{fp_fs:0.2f}"

    L = 2

    # Setup frontend
    analog_frontend_baseband = cbadc.synthesis.get_leap_frog(OSR=OSR * 2, N=N, T=1 / fs)

    fs = 1.0 / analog_frontend_baseband.digital_control.clock.T
    fp = fp_fs * fs

    T = 1 / fs

    M = 2 * N + 2

    # align fp
    period_samples = 4 * int(np.ceil(1.0 / (fp_fs)))
    size = target_size - (target_size % period_samples)
    size = target_size

    training_netlist = os.path.join(netlist_folder, f"rc_train_{unique_name}.cir")
    testing_netlist = os.path.join(netlist_folder, f"rc_test_{unique_name}.cir")

    training_ngspice_raw = os.path.join(ngspice_folder, f"rc_train_{unique_name}.raw")
    testing_ngspice_raw = os.path.join(ngspice_folder, f"rc_test_{unique_name}.raw")

    analog_frontend_bandpass = cbadc.synthesis.get_bandpass(
        analog_frontend=analog_frontend_baseband, fc=fp, phi=phi, delta_DC=delta_DC
    )

    # Modify for calibration
    Gamma = np.hstack(
        (np.zeros((2 * N, M - 2 * N)), analog_frontend_bandpass.analog_system.Gamma)
    )

    kappa_0 = kappa_0_scale * np.linalg.norm(Gamma[0:2, 2:4]) / np.sqrt(2)

    Gamma[0, 0] = kappa_0
    Gamma[N, 1] = kappa_0

    clock = cbadc.analog_signal.Clock(1 / fs, tt=1e-14, td=0.0)

    # add reference sequences
    analog_frontend_bandpass = cbadc.analog_frontend.AnalogFrontend(
        cbadc.analog_system.AnalogSystem(
            analog_frontend_bandpass.analog_system.A,
            analog_frontend_bandpass.analog_system.B,
            analog_frontend_bandpass.analog_system.CT,
            Gamma,
            analog_frontend_bandpass.analog_system.Gamma_tildeT,
        ),
        cbadc.digital_control.DitherControl(
            M - 2 * N,
            cbadc.digital_control.DigitalControl(
                clock,
                2 * N,
            ),
        ),
    )

    # input signal
    u_hat = np.zeros((L, size))

    frequency_baseband = 1.0 / analog_frontend_bandpass.digital_control.clock.T
    while frequency_baseband > BW / 3:
        frequency_baseband /= 2

    in_phase = cbadc.analog_signal.Sinusoidal(
        input_amplitude, frequency_baseband + fp, phase=np.pi / 2
    )
    quadrature = cbadc.analog_signal.Sinusoidal(
        input_amplitude, frequency_baseband + fp, phase=np.pi
    )

    # Make gmC circuit
    control_singals_filename = f"control_signals_{unique_name}.npy"
    vdd = 1.0
    C_int = 1e-12

    testbench_training = OpAmpTestBench(
        analog_frontend_bandpass,
        [
            cbadc.analog_signal.Sinusoidal(
                0.0,
                0.0,
            ),
            cbadc.analog_signal.Sinusoidal(
                0.0,
                0.0,
            ),
        ],
        clock,
        GBWP=GBWP,
        DC_gain=DC_gain,
        vdd_voltage=vdd,
        C_int=C_int,
        control_signal_vector_name=control_singals_filename,
    )

    testbench_testing = OpAmpTestBench(
        analog_frontend_bandpass,
        [quadrature, in_phase],
        clock,
        GBWP=GBWP,
        DC_gain=DC_gain,
        vdd_voltage=vdd,
        C_int=C_int,
        control_signal_vector_name=control_singals_filename,
    )

    # Training data simulation
    ngspice_simulator = NGSpiceSimulator(
        testbench_training,
        T,
        (training_size + warm_up) * T,
        netlist_filename=training_netlist,
        raw_output_filename=training_ngspice_raw,
    )
    ngspice_simulator.make_netlist()
    ngspice_simulator.run()
    ngspice_simulator.parse()

    control_signals = np.zeros((training_size, M))

    for _ in range(warm_up):
        next(ngspice_simulator)

    for index, s in enumerate(ngspice_simulator):
        control_signals[index, :] = s

    _, data = ngspice_simulator.get_state_trajectories()
    time_vector_training = data[::100, 0]
    states_training = data[::100, 1 : 1 + N].transpose()

    # initialize filter
    rel_BW = BW * T * 2 * 0.9
    print(f"BW: {rel_BW}")

    DSR = int(OSR >> 0)
    print(f"DSR: {DSR}")

    digital_estimator = cbadc.digital_estimator.adaptive_filter.AdaptiveFIRFilter(
        N, K, L=1, dtype=np.complex128
    )

    optimizer = {  # tf.keras.optimizers.SGD(
        "learning_rate": 1e-3,
        "momentum": 0.9,
    }  # )

    digital_estimator.compile(
        optimizer=optimizer,
    )

    modulation_freq = 1j * 2 * np.pi * fp * T

    modulation_sequence = np.exp(-modulation_freq * np.arange(control_signals.shape[0]))

    s_down_mixed = np.zeros((control_signals.shape[0], N + 1), dtype=np.complex128)

    for m in range(1, N + 1):
        s_down_mixed[:, m] = (
            2
            * (
                (control_signals[:, m + 1] - 0.5)
                + 1j * (control_signals[:, m + N + 1] - 0.5)
            )
            * modulation_sequence
        )
    s_down_mixed[:, 0] = (
        2
        * ((control_signals[:, 0] - 0.5) + 1j * (control_signals[:, 1] - 0.5))
        * modulation_sequence
    )

    s_decimated = scipy.signal.resample(
        s_down_mixed,
        control_signals.shape[0] // DSR,
        axis=0,
    )

    bw_rel = 0.5

    training_dataset_size = s_decimated.shape[0] - K
    K_0 = K - 8
    # bw_rel = DSR / OSR
    reference_filter = scipy.signal.firwin2(
        K_0, [0, bw_rel, bw_rel, 1], [1, 1 / np.sqrt(2), 0, 0]
    )
    from_index = (K >> 1) - (K_0 >> 1)
    to_index = (K >> 1) + (K_0 >> 1)

    s_dataset = np.zeros((training_dataset_size, M // 2, K), dtype=np.complex128)
    r_dataset = np.zeros((1, training_dataset_size), dtype=np.complex128)
    for i in range(training_dataset_size):
        for m in range(s_dataset.shape[1]):
            s_dataset[i, m, :] = s_decimated[i : i + K, m]
        r_dataset[0, i] = np.dot(reference_filter, s_dataset[i, 0, from_index:to_index])
    batch_size = 1 << 3

    digital_estimator.fit(
        x=s_dataset[:, 1:, :],
        y=r_dataset,
        batch_size=batch_size,
        epochs=epochs,
    )

    # Simulate test signal
    ngspice_simulator = NGSpiceSimulator(
        testbench_testing,
        T,
        (test_size) * T,
        netlist_filename=testing_netlist,
        raw_output_filename=testing_ngspice_raw,
    )
    ngspice_simulator.make_netlist()
    ngspice_simulator.run()
    ngspice_simulator.parse()

    time_vector = np.arange(test_size)
    states = np.zeros((2 * N, test_size))
    control_signals = np.zeros((test_size, M), dtype=np.float64)

    for index, s in enumerate(ngspice_simulator):
        control_signals[index, :] = s

    _, data = ngspice_simulator.get_state_trajectories()
    time_vector = data[::100, 0]
    states = data[::100, 1 : 1 + 2 * N].transpose()

    s_down_mixed = np.zeros((control_signals.shape[0], N + 1), dtype=np.complex128)

    for m in range(1, N + 1):
        s_down_mixed[:, m] = (
            2
            * (
                (control_signals[:, m + 1] - 0.5)
                + 1j * (control_signals[:, m + N + 1] - 0.5)
            )
            * modulation_sequence
        )
    s_down_mixed[:, 0] = (
        2
        * ((control_signals[:, 0] - 0.5) + 1j * (control_signals[:, 1] - 0.5))
        * modulation_sequence
    )

    s_decimated = scipy.signal.resample(
        s_down_mixed,
        control_signals.shape[0] // DSR,
        axis=0,
    )

    testing_dataset_size = s_decimated.shape[0] - K

    s_dataset = np.zeros((testing_dataset_size, M // 2, K), dtype=np.complex128)
    r_dataset = np.zeros((1, testing_dataset_size), dtype=np.complex128)
    for i in range(testing_dataset_size):
        for m in range(s_dataset.shape[1]):
            s_dataset[i, m, :] = s_decimated[i : i + K, m]
        r_dataset[0, i] = np.dot(reference_filter, s_dataset[i, 0, from_index:to_index])

    u_hat = np.real(
        digital_estimator.predict_full(s_dataset[:, 1:, :], r_dataset)
    ).flatten()

    # Plot the PSD of the digital estimator output.
    f, psd = cbadc.utilities.compute_power_spectral_density(
        u_hat[(K << 1) // DSR :],
        fs=1 / T / DSR,
        nperseg=testing_dataset_size - K,
    )
    signal_index = cbadc.utilities.find_sinusoidal(psd, 15)
    noise_index = np.ones(psd.size, dtype=bool)
    noise_index[signal_index] = False
    noise_index[f < (BW / 2.0)] = False
    fom = cbadc.utilities.snr_spectrum_computation_extended(
        psd, signal_index, noise_index, fs=1 / T / DSR
    )
    est_SNR = cbadc.fom.snr_to_dB(fom["snr"])
    est_ENOB = cbadc.fom.snr_to_enob(est_SNR)

    # Plot
    figure_filename = os.path.join(
        figures_folder,
        unique_name,
    )

    digital_estimator.plot_impulse_response()
    plt.savefig(f"{figure_filename}_imp.png")
    digital_estimator.plot_bode()
    plt.savefig(f"{figure_filename}_bode.png")

    # Save impulse response
    taps = np.arange(digital_estimator._h.shape[2])
    taps = taps.reshape((taps.size, 1))
    comb = np.hstack((taps, np.abs(digital_estimator._h[0, :, :].transpose())))

    np.savetxt(
        os.path.join(
            csv_folder,
            f"filter_plot_imp_op_amp_{digital_estimator._h.shape[1]}_{digital_estimator._h.shape[2]}.csv",
        ),
        comb,
        delimiter=",",
        fmt="%.6e",
        header=f"{','.join(['index'] +[f'h_{i}' for i in range(digital_estimator._h.shape[1])])}",
        comments="",
    )

    # Save bode plot data
    bode = np.fft.rfft(digital_estimator._h[0, :, :].transpose(), axis=0)
    freq = np.fft.rfftfreq(digital_estimator._h.shape[2])
    freq = freq.reshape((freq.size, 1))
    comb = np.hstack((freq, 20 * np.log10(np.abs(bode))))
    header = ",".join(["f"] + [f"h_{i}" for i in range(digital_estimator._h.shape[1])])
    np.savetxt(
        os.path.join(csv_folder, f"bode_{unique_name}.csv"),
        comb,
        delimiter=",",
        fmt="%.6e",
        header=header,
        comments="",
    )

    plt.figure()
    plt.title("Analog state vectors")
    for index in range(N):
        plt.plot(
            states[index, :],
            label="$x_{" + f"{index + 1}" + "}(t)$",
        )
    plt.grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
    plt.xlabel("$t/T$")
    plt.legend()

    plt.savefig(f"{figure_filename}.png")
    plt.close()
    plot_state_dist(states, f"{figure_filename}_dist.png")
    plt.close()

    plt.figure()
    for index in range(N):
        plt.plot(
            states_training[index, :],
            label="$x_{" + f"{index + 1}" + "}(t)$",
        )
    plt.grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
    plt.xlabel("$t/T$")
    plt.legend()

    plt.savefig(f"{figure_filename}_training.png")
    plt.close()

    plot_state_dist(states_training, f"{figure_filename}_training_dist.png")
    plt.close()

    plt.figure()
    for index in range(M)[::-1]:
        plt.plot(
            control_signals[:, index],
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
        label=f"N={N},SNR={est_SNR:.1f},OSR={OSR:.1f},amp={input_amplitude:.1f}",
    )
    psd_axis.plot(f, 10 * np.log10(noise_index * np.abs(psd)), label="noise index")

    psd_axis.legend(loc="lower left")
    psd_plot.savefig(f"{figure_filename}_psd.png")
    plt.close(psd_plot)

    time_plot = plt.figure()
    time_plot_axis = time_plot.add_subplot(111)
    time_plot_axis.set_title("Estimate")
    time_plot_axis.set_xlabel("t [samples]")
    time_plot_axis.set_ylabel("V")

    time_plot_axis.plot(u_hat)

    time_plot_axis.legend()
    time_plot.savefig(f"{figure_filename}_time.png")
    plt.close(time_plot)

    # Simulate state trajectories.
    result = {
        "state": {
            "time_vector": time_vector,
            "states": states,
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

    SNR = {}
    PSD = {}
    PSD_result = []
    PSD_header = []

    for index, res in cbadc.utilities.show_status(enumerate(results)):
        GBWP = res.args["rel_GBWP"]
        DC_gain = res.args["DC_gain"] / res.args["OSR"] * np.pi

        if DC_gain not in SNR:
            SNR[DC_gain] = []
        if DC_gain not in PSD:
            PSD[DC_gain] = []

        SNR[DC_gain].append((GBWP, res.res["psd"]["est_SNR"]))
        f, psd = cbadc.utilities.compute_power_spectral_density(
            res.res["u_hat"][res.args["K"] << 1 :],
            fs=res.args["fs"],
            nperseg=1 << 12,
        )
        PSD[DC_gain].append((GBWP, psd, f))

    # sort list
    for DC_gain in SNR:
        SNR[DC_gain] = sorted(SNR[DC_gain], key=lambda x: x[0])
        PSD[DC_gain] = sorted(PSD[DC_gain], key=lambda x: x[0])

    plt.figure()
    for DC_gain in SNR:
        SNR_header = []
        SNR_result = []
        for gwbp, psd, f in PSD[DC_gain]:
            PSD_result.append(f / res.args["fs"])
            PSD_result.append(10 * np.log10(np.abs(psd)))
            PSD_header.append(f"f_{DC_gain:.0f}_{gwbp:.0f}")
            PSD_header.append(f"PSD_{DC_gain:.0f}_{gwbp:.0f}")
        f_out, snr_out = outlier_remover(
            [x[0] for x in SNR[DC_gain]], [x[1] for x in SNR[DC_gain]]
        )
        SNR_result.append(f_out)
        SNR_result.append(snr_out)
        SNR_header.append("f")
        SNR_header.append("SNR")
        plt.semilogx(
            f_out,
            snr_out,
            "-*",
            label=f"DC_gain={DC_gain:.1e} " + "$\pi / \mathrm{OSR}$",
        )
        print(SNR_header)
        print(SNR_result)
        print(np.array(SNR_result).shape)
        np.savetxt(
            os.path.join(
                csv_folder,
                f"snr_op-amp_{DC_gain:.2e}_{res.args['N']}_{res.args['OSR']}.csv",
            ),
            np.array(SNR_result).transpose(),
            delimiter=",",
            fmt="%.6e",
            header=f"{','.join(SNR_header)}",
            comments="",
        )

    print(PSD_header)
    print(PSD_result)
    print(np.array(PSD_result).shape)
    np.savetxt(
        os.path.join(csv_folder, f"psd_op_amp_{res.args['N']}_{res.args['OSR']}.csv"),
        np.array(PSD_result).transpose(),
        delimiter=",",
        fmt="%.6e",
        header=f"{','.join(PSD_header)}",
        comments="",
    )

    plt.xlabel("$\mathrm{GBWP} / (f_p + \mathrm{BW} / 2)$")
    plt.ylabel("SNR")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(figures_folder, "GBWP_SNR.png"), dpi=dpi)
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
