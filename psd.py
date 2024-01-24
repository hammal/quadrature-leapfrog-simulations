from dataclasses import dataclass
import pickle
from typing import Any, Dict, Iterable
import logging
import simset
import numpy as np
import cbadc
import os
import matplotlib.pyplot as plt
from scipy import interpolate
from cbadc.analog_signal.quadrature import _rotation_matrix
import sys

logger = logging.getLogger(__name__)
plt.rcParams["figure.figsize"] = [10.0, 7.5]
K_ref_half = 1 << 9

simset.script_name = "psd.py"
simset.concurrent_jobs = 48
simset.python_interpreter = sys.executable


figures_folder = "figures/psd"
os.makedirs(figures_folder, exist_ok=True)

csv_folder = "csv/psd"
os.makedirs(csv_folder, exist_ok=True)


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


# sampling frequency
fs = 1 << 31

# Set length and oversampling ratio

N = 8
OSR = 4

# N = 6
# OSR = 4

# N = 6
# OSR = 8

# number of equally spaced notch frequency points
number_of_freq_points = OSR

print(OSR)


#######################################################################################
# The simulate function to be executed at each node.
#######################################################################################
@simset.arg("N", [N])
@simset.arg("fs", [fs])
@simset.arg("OSR", [OSR])
@simset.arg(
    "fp_fs",
    1 / (4 * OSR)
    + np.arange(number_of_freq_points)
    / float(number_of_freq_points - 1)
    * (0.5 - 1 / (2 * OSR)),
)
@simset.arg("input_amplitude", [1.0])
@simset.arg("target_size", [1 << 14])
@simset.arg("K1", [1 << 11])
@simset.arg("K2", [1 << 11])
@simset.arg("phi", [0.01])
@simset.arg("delta_DC", [0.0])
@simset.arg("kappa_0_scale", [0.1])
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
):
    """
    The simulate function which will run on each computer node
    with an unique argument tuple. Note that the order of the
    @simset.arg(...) and simulate_function(arg,...) must match.
    """
    logger.info("Starting simulation.")
    # relative bandwidth
    L = 2

    logger.info(f"fp_fs: {fp_fs}")
    BW = 0.5 * fs / OSR

    # Setup baseband analog frontend
    analog_frontend_baseband = cbadc.synthesis.get_leap_frog(OSR=2 * OSR, N=N, T=1 / fs)

    fs = 1.0 / analog_frontend_baseband.digital_control.clock.T
    fp = fp_fs * fs

    # align fp
    period_samples = 4 * int(np.ceil(1.0 / (fp_fs)))
    size = target_size - (target_size % period_samples)
    size = target_size

    args = {
        "BW": BW,
        "OSR": OSR,
        "N": N,
        "input_amplitude": input_amplitude,
        "fp_fs": fp_fs,
        "fp": fp,
        "fs": fs,
        "size": size,
        "K1": K1,
        "K2": K2,
        "phi": phi,
        "delta_DC": delta_DC,
        "size": size,
    }

    logging.info(args)
    # setup quadrature analog frontend
    analog_frontend_bandpass = cbadc.synthesis.get_bandpass(
        analog_frontend=analog_frontend_baseband, fc=fp, phi=phi, delta_DC=delta_DC
    )

    # Modify for calibration
    Gamma = np.hstack(
        (np.zeros((2 * N, 2)), analog_frontend_bandpass.analog_system.Gamma)
    )

    kappa_0 = kappa_0_scale * np.linalg.norm(Gamma[0:2, 2:4]) / np.sqrt(2)

    Gamma[0, 0] = kappa_0
    Gamma[N, 1] = kappa_0

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
            2,
            cbadc.digital_control.DigitalControl(
                cbadc.analog_signal.Clock(1 / fs, tt=1e-14, td=0.0),
                2 * N,
                impulse_response=analog_frontend_bandpass.digital_control._impulse_response[
                    0
                ],
            ),
            impulse_response=analog_frontend_bandpass.digital_control._impulse_response[
                0
            ],
        ),
    )

    impulse_response_time = np.arange(100) / 100 * 2 / fs
    impulse_response = np.array(
        [
            analog_frontend_bandpass.digital_control.impulse_response(0, _t)
            for _t in impulse_response_time
        ]
    )

    logger.info(analog_frontend_bandpass)

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

    digital_estimator_baseband = cbadc.digital_estimator.FIRFilter(
        analog_frontend_baseband.analog_system,
        analog_frontend_baseband.digital_control,
        eta2,
        K1,
        K2,
    )

    logger.info(digital_estimator)

    u_hat = np.zeros((L, size))
    u_hat_dm = np.zeros_like(u_hat)

    frequency_baseband = 1.0 / analog_frontend_bandpass.digital_control.clock.T
    while frequency_baseband > BW / 3:
        frequency_baseband /= 2

    frequency = 1.0 / analog_frontend_bandpass.digital_control.clock.T
    while frequency > BW / 2 + fp or frequency < fp - BW / 2:
        if frequency > BW / 2 + fp + 1e-3:
            frequency /= 2
        else:
            frequency = fp  # - 0.5 * BW
    frequency = fp

    analog_frontend_bandpass.digital_control.reset()
    quadrature_input_signal_pair = cbadc.analog_signal.get_quadrature_signal_pair(
        amplitude=1.0,
        angular_frequency=2 * np.pi * fp,
        in_phase=cbadc.analog_signal.Sinusoidal(
            input_amplitude, frequency_baseband, phase=np.pi / 2
        ),
        quadrature=cbadc.analog_signal.Sinusoidal(
            input_amplitude, frequency_baseband, phase=np.pi
        ),
    )
    simulator = cbadc.simulator.get_simulator(
        analog_frontend_bandpass.analog_system,
        analog_frontend_bandpass.digital_control,
        quadrature_input_signal_pair,
        simulator_type=cbadc.simulator.SimulatorType.full_numerical,
    )
    ext_simulator = cbadc.simulator.extended_simulation_result(simulator)

    logger.info(simulator)

    logger.info(analog_frontend_bandpass.digital_control._impulse_response[0])

    digital_estimator(simulator)

    state_sim_size = 1 << 10
    states = np.zeros((2 * N, state_sim_size))

    for index in cbadc.utilities.show_status(range(state_sim_size)):
        res = next(ext_simulator)
        states[:, index] = res["analog_state"]

    angle = np.zeros(size)
    for index in cbadc.utilities.show_status(range(size)):
        u_hat[:, index] = next(digital_estimator)
        u_hat_dm[:, index] = np.dot(
            _rotation_matrix(-2 * np.pi * fp * (index - K1 - K2) / fs), u_hat[:, index]
        )
        angle[index] = np.arctan2(u_hat_dm[1, index], u_hat_dm[0, index])

    bandpass_res = evaluate(u_hat[:, K1 + K2 :], fp, fs, BW / 2, size)
    u_hat_combined = (u_hat[0, K1 + K2 :] + u_hat[1, K1 + K2 :]) / np.sqrt(2)
    bandpass_res_2 = evaluate(
        np.stack((u_hat_combined, u_hat_combined), axis=0), fp, fs, BW / 2, size
    )
    bandpass_dm_res_0 = evaluate_baseband(u_hat_dm[0, K1 + K2 :], fs, BW / 2, size)
    bandpass_dm_res_1 = evaluate_baseband(u_hat_dm[1, K1 + K2 :], fs, BW / 2, size)

    u_hat_baseband = np.zeros(size)

    analog_frontend_baseband.digital_control.reset()
    input_signal_BB = cbadc.analog_signal.Sinusoidal(
        input_amplitude, frequency_baseband
    )

    simulator_baseband = cbadc.simulator.get_simulator(
        analog_frontend_baseband.analog_system,
        analog_frontend_baseband.digital_control,
        [input_signal_BB],
        simulator_type=cbadc.simulator.SimulatorType.pre_computed_numerical,
    )

    digital_estimator_baseband(simulator_baseband)
    for index in cbadc.utilities.show_status(range(size)):
        u_hat_baseband[index] = next(digital_estimator_baseband)
    baseband_res = evaluate_baseband(u_hat_baseband[K1 + K2 :], fs, BW / 2, size)

    # put the results in the ResultDataClass
    logger.info(f"OSR = {OSR}")
    result_dataclass = ResultDataClass(
        args=args,
        res={
            "bandpass": bandpass_res,
            "bandpass_filter": digital_estimator,
            "impulse_response": impulse_response,
            "impulse_response_time": impulse_response_time,
            "baseband_res": baseband_res,
            "bandpass_dm_res_0": bandpass_dm_res_0,
            "bandpass_dm_res_1": bandpass_dm_res_1,
            "bandpass_2": bandpass_res_2,
            "angle": angle,
            "states": states,
        },
        time={},
    )

    # return the results dataclass and it will automatically be stored
    return result_dataclass


TIME_PLOTS = True
PSD_PLOTS = True


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

    f = []
    snr_0 = []
    snr_1 = []
    snr_bb = []
    snr_dm_0 = []
    snr_dm_1 = []

    f_psd, ax_psd = plt.subplots(2, 1, sharey=True)
    f_psd_dm, ax_psd_dm = plt.subplots(2, 1, sharey=True)
    f_angle, ax_angle = plt.subplots(1, 1, sharey=True)
    f_states, ax_states = plt.subplots()

    psd_results = []
    psd_results_header = []

    for index, res in cbadc.utilities.show_status(enumerate(results)):
        BW = res.args["BW"]
        OSR = res.args["OSR"]
        N = res.args["N"]
        fs = res.args["fs"]
        fp = res.args["fp"]
        size = res.args["size"]
        logger.info(f"fp: {fp}, fs: {fs}, BW: {BW}, OSR: {OSR}, N: {N}")

        f_states, ax_states = plt.subplots()
        for index_x in range(2 * N):
            ax_states.plot(
                res.res["states"][index, :], label="$x_{" + str(index_x + 1) + "}(t)$"
            )
        ax_states.legend()
        ax_states.grid()
        ax_states.set_xlim(0, 100)
        ax_states.set_xlabel("t")
        f_states.savefig(
            os.path.join(
                figures_folder,
                f"states_{index}_OSR_{res.args['OSR']}_N_{res.args['N']}.png",
            ),
            dpi=dpi,
        )
        plt.close(f_states)

        f.append(res.args["fp"])
        snr_0.append(res.res["bandpass"]["est_SNR_0"])
        snr_1.append(res.res["bandpass"]["est_SNR_1"])

        snr_bb.append(res.res["baseband_res"]["est_SNR"])

        snr_dm_0.append(res.res["bandpass_dm_res_0"]["est_SNR"])
        snr_dm_1.append(res.res["bandpass_dm_res_1"]["est_SNR"])

        if TIME_PLOTS:
            # Time Domain
            f_time, ax_time = plt.subplots(1, 1, sharex=True)
            f_time_2, ax_time_2 = plt.subplots(1, 1, sharex=True)
            ax_time.plot(
                res.res["bandpass_2"]["t"][:1000],
                res.res["bandpass_2"]["u_hat_0"][:1000],
                label=f"u+u_b, ENOB: {res.res['bandpass_2']['est_ENOB_0']:.2f}",
            )
            ax_time.plot(
                res.res["bandpass"]["t"][:1000],
                res.res["bandpass"]["u_hat_0"][:1000],
                label=f"u, ENOB: {res.res['bandpass']['est_ENOB_0']:.2f}",
            )
            ax_time.plot(
                res.res["bandpass"]["t"][:1000],
                res.res["bandpass"]["u_hat_1"][:1000],
                label=f"u_b, ENOB: {res.res['bandpass']['est_ENOB_1']:.2f}",
            )

            ax_time_2.plot(
                res.res["bandpass_dm_res_0"]["t"][:1000],
                res.res["bandpass_dm_res_0"]["u_hat"][:1000],
                label=f"DM, Q, ENOB: {res.res['bandpass_dm_res_0']['est_ENOB']:.2f}",
            )
            ax_time_2.plot(
                res.res["bandpass_dm_res_1"]["t"][:1000],
                res.res["bandpass_dm_res_1"]["u_hat"][:1000],
                label=f"DM, I, ENOB: {res.res['bandpass_dm_res_1']['est_ENOB']:.2f}",
            )
            ax_time.legend()
            ax_time.set_title(f"Waveform in time, fp={res.args['fp']:0.1e} Hz")
            ax_time.set_xlabel("time [s]")
            ax_time.set_ylabel("$\hat{u}(.)$")
            ax_time.grid(True)

            ax_time_2.legend()
            ax_time_2.set_title(f"Waveform in time, fp={res.args['fp']:0.1e} Hz")
            ax_time_2.set_xlabel("time [s]")
            ax_time_2.set_ylabel("$\hat{u}(.)$")
            ax_time_2.grid(True)

            f_time.savefig(
                os.path.join(
                    figures_folder,
                    f"time_domain_{index}_{res.args['N']}_{res.args['OSR']}.png",
                ),
                dpi=dpi,
            )
            plt.close(f_time)
            f_time_2.savefig(
                os.path.join(
                    figures_folder,
                    f"time_domain_dm_{index}_{res.args['N']}_{res.args['OSR']}.png",
                ),
                dpi=dpi,
            )
            plt.close(f_time_2)

        freq, Pxx_den_0 = res.res["bandpass"]["f"], res.res["bandpass"]["psd_0"]
        npoints = 1 << 11
        fmin = np.min(freq)
        fmax = np.max(freq)
        f_pxx = interpolate.interp1d(freq, Pxx_den_0)
        freq = np.linspace(fmin, fmax, int(npoints))
        # freq = np.geomspace(1e-1, fmax, int(npoints))
        Pxx_den_0 = f_pxx(freq)

        freq, Pxx_den_0 = cbadc.utilities.compute_power_spectral_density(
            res.res["bandpass"]["u_hat_0"],
            fs=fs,
            nperseg=npoints,
        )
        freq, Pxx_den_1 = cbadc.utilities.compute_power_spectral_density(
            res.res["bandpass"]["u_hat_1"],
            fs=fs,
            nperseg=npoints,
        )

        freq, Pxx_den_0_1 = cbadc.utilities.compute_power_spectral_density(
            res.res["bandpass_2"]["u_hat_0"],
            fs=fs,
            nperseg=npoints,
        )

        freq, Pxx_dm_0 = cbadc.utilities.compute_power_spectral_density(
            res.res["bandpass_dm_res_0"]["u_hat"],
            fs=fs,
            nperseg=npoints,
        )
        freq, Pxx_dm_1 = cbadc.utilities.compute_power_spectral_density(
            res.res["bandpass_dm_res_1"]["u_hat"],
            fs=fs,
            nperseg=npoints,
        )

        if PSD_PLOTS:
            # PSD
            ax_psd[1].semilogx(
                freq,
                10 * np.log10(Pxx_den_0),
                label=f"u, fp: {res.args['fp']:0.1e} Hz, SNR: {res.res['bandpass']['est_SNR_0']:.1f} dB",
            )
            ax_psd[0].plot(
                freq,
                10 * np.log10(Pxx_den_0),
                label=f"u, fp: {res.args['fp']:0.1e} Hz, SNR: {res.res['bandpass']['est_SNR_0']:.1f} dB",
            )
            ax_psd[1].semilogx(
                freq,
                10 * np.log10(Pxx_den_1),
                label=f"u_b, fp: {res.args['fp']:0.1e} Hz, SNR: {res.res['bandpass']['est_SNR_1']:.1f} dB",
            )
            ax_psd[0].plot(
                freq,
                10 * np.log10(Pxx_den_1),
                label=f"u_b, fp: {res.args['fp']:0.1e} Hz, SNR: {res.res['bandpass']['est_SNR_1']:.1f} dB",
            )

            # Both
            ax_psd[1].semilogx(
                freq,
                10 * np.log10(Pxx_den_0_1),
                label=f"u+u_b, fp: {res.args['fp']:0.1e} Hz, SNR: {res.res['bandpass_2']['est_SNR_0']:.1f} dB",
            )
            ax_psd[0].plot(
                freq,
                10 * np.log10(Pxx_den_0_1),
                label=f"u+u_b, fp: {res.args['fp']:0.1e} Hz, SNR: {res.res['bandpass_2']['est_SNR_0']:.1f} dB",
            )

            # DM PSD
            ax_psd_dm[1].semilogx(
                freq,
                10 * np.log10(Pxx_dm_0),
                label=f"DM, Q, fp: {res.args['fp']:0.1e} Hz, SNR: {res.res['bandpass_dm_res_0']['est_SNR']:.1f} dB",
            )
            ax_psd_dm[0].plot(
                freq,
                10 * np.log10(Pxx_dm_0),
                label=f"DM, Q, fp: {res.args['fp']:0.1e} Hz, SNR: {res.res['bandpass_dm_res_0']['est_SNR']:.1f} dB",
            )
            ax_psd_dm[1].semilogx(
                freq,
                10 * np.log10(Pxx_dm_1),
                label=f"DM, I, fp: {res.args['fp']:0.1e} Hz, SNR: {res.res['bandpass_dm_res_1']['est_SNR']:.1f} dB",
            )
            ax_psd_dm[0].plot(
                freq,
                10 * np.log10(Pxx_dm_1),
                label=f"DM, I, fp: {res.args['fp']:0.1e} Hz, SNR: {res.res['bandpass_dm_res_1']['est_SNR']:.1f} dB",
            )

            # ax_psd[0].plot(
            #     res.res["bandpass"]["f"],
            #     10 * np.log10(res.res["bandpass"]["psd_0"]),
            #     label=f"fp: {res.args['fp']:0.1e} Hz, SNR: {res.res['bandpass']['est_SNR_0']:.1f} dB",
            # )

        psd_results.append(freq / fs)
        psd_results.append(10 * np.log10(Pxx_den_0))
        psd_results_header.append(f"f_fp_{index}")
        psd_results_header.append(f"fp_{index}")

        ax_angle.plot(res.res["angle"][:200], label=f'fp: {res.args["fp"]:0.1e} Hz')

    ax_angle.set_xlabel("Time index")
    ax_angle.set_ylabel("Angle [rad]")
    ax_angle.legend()
    f_angle.savefig(os.path.join(figures_folder, f"angle.png"), dpi=dpi)
    plt.close(f_angle)

    freq, Pxx_den_0 = cbadc.utilities.compute_power_spectral_density(
        res.res["baseband_res"]["u_hat"],
        fs=fs,
        nperseg=npoints,
    )
    psd_results.append(freq / fs)
    psd_results.append(10 * np.log10(Pxx_den_0))
    psd_results_header.append(f"f_bb")
    psd_results_header.append(f"bb")

    if TIME_PLOTS:
        f_time, ax_time = plt.subplots(1, 1, sharex=True)
        ax_time.plot(
            res.res["baseband_res"]["t"][:1000],
            res.res["baseband_res"]["u_hat"][:1000],
            label=f"bandpass, ENOB: {res.res['baseband_res']['est_ENOB']:.2f}",
        )
        ax_time.legend()
        ax_time.set_title(f"Waveform in time, fp={res.args['fp']:0.1e} Hz")
        ax_time.set_xlabel("time [s]")
        ax_time.set_ylabel("$\hat{u}(.)$")
        ax_time.grid(True)

        f_time.savefig(
            os.path.join(
                figures_folder,
                f"time_domain_{index}_{res.args['N']}_{res.args['OSR']}_bb.png",
            ),
            dpi=dpi,
        )
        plt.close(f_time)

    if PSD_PLOTS:
        ax_psd[0].plot(
            freq,
            10 * np.log10(Pxx_den_0),
            "--",
            label=f"bb, SNR: {res.res['baseband_res']['est_SNR']:.1f} dB",
        )
        ax_psd[1].plot(
            freq,
            10 * np.log10(Pxx_den_0),
            "--",
            label=f"bb, SNR: {res.res['baseband_res']['est_SNR']:.1f} dB",
        )
        ax_psd_dm[0].plot(
            freq,
            10 * np.log10(Pxx_den_0),
            label=f"bb, SNR: {res.res['baseband_res']['est_SNR']:.1f} dB",
        )
        ax_psd_dm[1].plot(
            freq,
            10 * np.log10(Pxx_den_0),
            label=f"bb, SNR: {res.res['baseband_res']['est_SNR']:.1f} dB",
        )

    ax_psd[1].legend(loc="lower left")
    ax_psd[0].set_title(
        f"PSDs for BW: {BW:0.1e} Hz, OSR: {res.args['OSR']}, N: {res.args['N']}, fs: {res.args['fs']:0.1e} Hz"
    )
    ax_psd[0].set_ylabel("dB")
    ax_psd[0].grid(True)
    ax_psd[1].set_ylabel("dB")
    ax_psd[1].set_xlabel("Frequency [Hz]")
    ax_psd[1].grid(True)
    ax_psd[1].set_xlim((fs / 2 * 1e-3, fs / 2))
    ax_psd[0].set_xlim((0, fs / 2))

    f_psd.savefig(
        os.path.join(figures_folder, f"psd_{res.args['N']}_{res.args['OSR']}.png"),
        dpi=dpi,
    )
    plt.close(f_psd)

    ax_psd_dm[1].legend(loc="lower left")
    ax_psd_dm[0].set_title(
        f"PSDs for BW: {BW:0.1e} Hz, OSR: {res.args['OSR']}, N: {res.args['N']}, fs: {res.args['fs']:0.1e} Hz"
    )
    ax_psd_dm[0].set_ylabel("dB")
    ax_psd_dm[0].grid(True)
    ax_psd_dm[1].set_ylabel("dB")
    ax_psd_dm[1].set_xlabel("Frequency [Hz]")
    ax_psd_dm[1].grid(True)
    ax_psd_dm[1].set_xlim((fs / 2 * 1e-3, fs / 2))
    ax_psd_dm[0].set_xlim((0, fs / 2))

    f_psd_dm.savefig(
        os.path.join(figures_folder, f"psd_dm_{res.args['N']}_{res.args['OSR']}.png"),
        dpi=dpi,
    )
    plt.close(f_psd_dm)

    snr_l2 = (
        10 * np.log10(np.linalg.norm(np.power(10, np.array(x) / 10)))
        for x in zip(snr_0, snr_1)
    )

    f_snr_0, snr_0 = (
        list(x) for x in zip(*sorted(zip(f, snr_0), key=lambda pair: pair[0]))
    )
    f_snr_1, snr_1 = (
        list(x) for x in zip(*sorted(zip(f, snr_1), key=lambda pair: pair[0]))
    )
    f_snr_l2, snr_l2 = (
        list(x) for x in zip(*sorted(zip(f, snr_l2), key=lambda pair: pair[0]))
    )

    snr_dm_l2 = (
        10 * np.log10(np.linalg.norm(np.power(10, np.array(x) / 10)))
        for x in zip(snr_dm_0, snr_dm_1)
    )

    f_snr_dm_0, snr_dm_0 = (
        list(x) for x in zip(*sorted(zip(f, snr_dm_0), key=lambda pair: pair[0]))
    )
    f_snr_dm_1, snr_dm_1 = (
        list(x) for x in zip(*sorted(zip(f, snr_dm_1), key=lambda pair: pair[0]))
    )
    f_snr_dm_l2, snr_dm_l2 = (
        list(x) for x in zip(*sorted(zip(f, snr_dm_l2), key=lambda pair: pair[0]))
    )

    f_snr, ax_snr = plt.subplots(2, 1)
    f_snr_2, ax_snr_2 = plt.subplots(2, 1)
    f_snr_3, ax_snr_3 = plt.subplots(2, 1)
    ax_snr[0].plot(np.array(f_snr_0) / fs, snr_0, "-o", label="SNR_u")
    ax_snr[1].semilogx(np.array(f_snr_0) / fs, snr_0, "-o", label="SNR_u")
    ax_snr[0].plot(np.array(f_snr_1) / fs, snr_1, "-o", label="SNR_u_b")
    ax_snr[1].semilogx(np.array(f_snr_1) / fs, snr_1, "-o", label="SNR_u_b")

    ax_snr_2[0].plot(np.array(f_snr_dm_0) / fs, snr_dm_0, "-o", label="SNR_Q")
    ax_snr_2[1].semilogx(np.array(f_snr_dm_0) / fs, snr_dm_0, "-o", label="SNR_Q")
    ax_snr_2[0].plot(np.array(f_snr_dm_1) / fs, snr_dm_1, "-o", label="SNR_I")
    ax_snr_2[1].semilogx(np.array(f_snr_dm_1) / fs, snr_dm_1, "-o", label="SNR_I")

    ax_snr_3[0].plot(np.array(f_snr_dm_l2) / fs, snr_dm_l2, "-o", label="Q+I_SNR_L2")
    ax_snr_3[1].semilogx(
        np.array(f_snr_dm_l2) / fs, snr_dm_l2, "-o", label="Q+I_SNR_L2"
    )
    ax_snr_2[0].plot(np.array(f_snr_dm_l2) / fs, snr_dm_l2, "-o", label="Q+I_SNR_L2")
    ax_snr_2[1].semilogx(
        np.array(f_snr_dm_l2) / fs, snr_dm_l2, "-o", label="Q+I_SNR_L2"
    )

    ax_snr_3[0].plot(np.array(f_snr_l2) / fs, snr_l2, "-o", label="u+u_bar_SNR_L2")
    ax_snr_3[1].semilogx(np.array(f_snr_l2) / fs, snr_l2, "-o", label="u+u_bar_SNR_L2")
    ax_snr[0].plot(np.array(f_snr_l2) / fs, snr_l2, "-o", label="u+u_bar_SNR_L2")
    ax_snr[1].semilogx(np.array(f_snr_l2) / fs, snr_l2, "-o", label="u+u_bar_SNR_L2")

    ax_snr[0].plot(np.array(f_snr_0) / fs, snr_bb, "-o", label="SNR_BB")
    ax_snr[1].semilogx(np.array(f_snr_0) / fs, snr_bb, "-o", label="SNR_BB")
    ax_snr_2[0].plot(np.array(f_snr_0) / fs, snr_bb, "-o", label="SNR_BB")
    ax_snr_2[1].semilogx(np.array(f_snr_0) / fs, snr_bb, "-o", label="SNR_BB")
    ax_snr_3[0].plot(np.array(f_snr_0) / fs, snr_bb, "-o", label="SNR_BB")
    ax_snr_3[1].semilogx(np.array(f_snr_0) / fs, snr_bb, "-o", label="SNR_BB")

    ax_snr[0].set_xlabel("fp/fs")
    ax_snr[0].set_ylabel("SNR [dB]")
    ax_snr[1].set_xlabel("fp")
    ax_snr[1].set_ylabel("SNR [dB]")
    ax_snr[1].grid(True)
    ax_snr[0].grid(True)
    ax_snr[1].legend()
    f_snr.savefig(
        os.path.join(figures_folder, f"snr_{res.args['N']}_{res.args['OSR']}.png"),
        dpi=dpi,
    )
    plt.close(f_snr)

    ax_snr_2[0].set_xlabel("fp/fs")
    ax_snr_2[0].set_ylabel("SNR [dB]")
    ax_snr_2[1].set_xlabel("fp")
    ax_snr_2[1].set_ylabel("SNR [dB]")
    ax_snr_2[1].grid(True)
    ax_snr_2[0].grid(True)
    ax_snr_2[1].legend()
    f_snr_2.savefig(
        os.path.join(figures_folder, f"snr_2_{res.args['N']}_{res.args['OSR']}.png"),
        dpi=dpi,
    )
    plt.close(f_snr)

    ax_snr_3[0].set_xlabel("fp/fs")
    ax_snr_3[0].set_ylabel("SNR [dB]")
    ax_snr_3[1].set_xlabel("fp")
    ax_snr_3[1].set_ylabel("SNR [dB]")
    ax_snr_3[1].grid(True)
    ax_snr_3[0].grid(True)
    ax_snr_3[1].legend()
    f_snr_3.savefig(
        os.path.join(figures_folder, f"snr_3_{res.args['N']}_{res.args['OSR']}.png"),
        dpi=dpi,
    )
    plt.close(f_snr)

    # Check impulse responses
    f_imp, ax_imp = plt.subplots(1)
    ax_imp.plot(res.res["impulse_response_time"] * fs, res.res["impulse_response"])
    ax_imp.set_xlabel("t/T")
    ax_imp.set_ylabel("impulse response")
    ax_imp.grid(True)
    f_imp.savefig(os.path.join(figures_folder, "impulse_response.png"), dpi=dpi)
    plt.close(f_imp)

    # plt.show()

    psd_results = np.stack(psd_results, axis=1)
    np.savetxt(
        os.path.join(csv_folder, f"psd_{res.args['N']}_{res.args['OSR']}.csv"),
        np.array(psd_results),
        delimiter=",",
        fmt="%.6e",
        header=f"{','.join(psd_results_header)}",
        comments="",
    )
    np.savetxt(
        os.path.join(csv_folder, f"snr_{res.args['N']}_{res.args['OSR']}.csv"),
        np.array(
            [np.array(f_snr_0) / fs, np.array(f_snr_1) / fs, snr_0, snr_1]
        ).transpose(),
        delimiter=",",
        fmt="%.6e",
        header=f"f_I,f_Q,SNR_I,SNR_Q",
        comments="",
    )
    np.savetxt(
        os.path.join(csv_folder, f"snr_bb.csv"),
        np.array([snr_bb]).transpose(),
        delimiter=",",
        fmt="%.6e",
        header=f"SNR_BB",
        comments="",
    )


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
