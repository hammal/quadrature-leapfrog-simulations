import subprocess
from typing import List
import os
import numpy as np
import cbadc
import matplotlib.pyplot as plt
from typing import Any, Dict, Iterable
import pickle


plt.rcParams["agg.path.chunksize"] = 10000

psd_size = 1 << 18
dpi = 1200


class Calib:
    def __init__(self, filter_name, BW, M, K=1 << 10, k0=0.1, path="."):
        self.path = path
        filter_name = os.path.join(
            self.path, os.path.basename(os.path.splitext(filter_name)[0])
        )
        # print(filter_name)
        self.filter_name = filter_name + ".npy"
        # print(self.filter_name)
        self.BW = BW
        self.M = M
        self.K = K
        self.k0 = k0
        self.unique_name = f"{filter_name}_{self.BW:0.3f}_{self.M}_{self.K}_{self.k0}"
        # print(self.unique_name)
        self._run(
            [
                "calib_filter",
                "create",
                "-bw",
                str(self.BW),
                "-m",
                str(self.M),
                "-k",
                str(K),
                "-k0s",
                str(self.k0),
                self.filter_name,
            ],
        )

    def calibrate(
        self,
        training_control_signals,
        batch_size=1 << 8,
        step_size=1e-6,
        decay=0.9,
        training_iterations=1 << 20,
    ):
        self.calibration_output = f"{self.unique_name}_calibrate.npy"
        self._run(
            [
                "calib",
                "calibrate",
                "-i",
                training_control_signals,
                "--batch-size",
                str(batch_size),
                "-f",
                self.filter_name,
                "-s",
                str(step_size),
                "-d",
                str(decay),
                "--iterations",
                str(training_iterations),
                "-o",
                self.calibration_output,
            ]
        )
        filter_res = np.load(self.filter_name)
        filter_name = os.path.splitext(self.filter_name)[0]
        plot_impulse_response(filter_res, f"{filter_name}_imp.png")
        bode_plot(filter_res, f"{filter_name}_bode.png")
        self.plot_result(self.calibration_output)

    def plot_result(self, result_file, train=True):
        # plot results
        u_hat = np.load(result_file)
        basename = (
            f"{os.path.splitext(result_file)[0]}_{'train' if train else 'validate'}"
        )
        plot_psd(u_hat, f"{basename}_psd.png", self.BW)
        plot_time_domain(u_hat, f"{basename}_time_domain.png")

    def validate(self, validation_control_signals):
        self.validation_output = f"{self.unique_name}_validate.npy"
        self._run(
            [
                "calib",
                "validate",
                "-i",
                validation_control_signals,
                "-f",
                self.filter_name,
                "-o",
                self.validation_output,
            ]
        )
        self.plot_result(self.validation_output, train=False)

    def _run(self, command: List[str]):
        subprocess.run(command, capture_output=True)


def psd_evaluate(u_hat: np.ndarray, fs: float, BW: float, psd_size: int = 1 << 18):
    f, psd = cbadc.utilities.compute_power_spectral_density(
        u_hat,
        fs=fs,
        nperseg=min(psd_size, u_hat.size),
    )
    signal_index = cbadc.utilities.find_sinusoidal(psd, 25)
    # print(signal_index)
    # psd_harm = psd
    # psd_harm[signal_index] = psd[:int(np.mean(signal_index)) - 15].mean()
    # harmonics_mask = cbadc.utilities.find_n_sinusoidals(psd_harm, 1, 25)
    noise_index = np.ones(psd.size, dtype=bool)
    noise_index[signal_index] = False
    noise_index[f < (BW * 1e-2)] = False
    noise_index[f > BW] = False
    # print(harmonics_mask)
    fom = cbadc.utilities.snr_spectrum_computation_extended(
        psd,
        signal_index,
        noise_index,
        # harmonics_mask,
        fs=fs,
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


def plot_psd(u_hat: np.ndarray, figure_path: str, BW: float, linear: bool = False):
    # u_hat = np.load(data_path)
    res = psd_evaluate(u_hat, 1.0, BW, psd_size)
    f_psd, ax_psd = plt.subplots(1, 1, sharex=True)
    if linear:
        ax_psd.plot(
            res["f"],
            10 * np.log10(res["psd"]),
            label=f"SNR: {res['est_SNR']:.2f} dB",
        )
    else:
        ax_psd.semilogx(
            res["f"],
            10 * np.log10(res["psd"]),
            label=f"SNR: {res['est_SNR']:.2f} dB",
        )
    ax_psd.legend()
    ax_psd.set_title("power spectral density (PSD)")
    ax_psd.set_xlabel("frequency")
    ax_psd.set_ylabel("dB")
    ax_psd.grid(True)
    f_psd.savefig(
        figure_path,
        dpi=dpi,
    )
    plt.close(f_psd)


def plot_impulse_response(h: np.ndarray, figure_path: str):
    # h = np.load(filter_path)
    f_h, ax_h = plt.subplots(2, 1, sharex=True)
    # L = h.shape[0]
    # K = h.shape[2]
    M = h.shape[1]
    for m in range(M):
        h_version = h[0, m, :]
        ax_h[0].plot(
            np.arange(h_version.size) - h_version.size // 2,
            h_version,
            label="$h_{" + f"{m}" + "}$",
        )
        ax_h[1].semilogy(
            np.arange(h_version.size) - h_version.size // 2,
            np.abs(h_version),
            label="$h_{" + f"{m}" + "}$",
        )

    ax_h[0].legend()
    ax_h[0].set_title("impulse responses")
    ax_h[1].set_xlabel("filter taps")
    ax_h[0].set_ylabel("$h[.]$")
    ax_h[1].set_ylabel("$|h[.]|$")
    ax_h[0].grid(True)
    ax_h[1].grid(True)
    f_h.savefig(
        figure_path,
        dpi=dpi,
    )
    plt.close(f_h)


def bode_plot(h: np.ndarray, figure_path: str, linear: bool = False):
    f_h, ax_h = plt.subplots(2, 1, sharex=True)
    M = h.shape[1]
    for m in range(M):
        h_version = h[0, m, :]

        h_freq = np.fft.rfft(h_version)
        freq = np.fft.rfftfreq(h_version.size)

        if linear:
            ax_h[1].plot(
                freq,
                np.angle(h_freq),
                label="$h_{" + f"{m}" + "}$",
            )
            ax_h[0].plot(
                freq,
                20 * np.log10(np.abs(h_freq)),
                label="$h_{" + f"{m}" + "}$",
            )
        else:
            ax_h[1].semilogx(
                freq,
                np.angle(h_freq),
                label="$h_{" + f"{m}" + "}$",
            )
            ax_h[0].semilogx(
                freq,
                20 * np.log10(np.abs(h_freq)),
                label="$h_{" + f"{m}" + "}$",
            )

    ax_h[0].legend()
    ax_h[0].set_title("Bode diagram")
    ax_h[1].set_xlabel("frequency [Hz]")
    ax_h[1].set_ylabel("$ \\angle h[.]$ rad")
    ax_h[0].set_ylabel("$|h[.]|$ dB")
    ax_h[0].grid(True)
    ax_h[1].grid(True)
    f_h.savefig(
        figure_path,
        dpi=dpi,
    )
    plt.close(f_h)


def plot_time_domain(y: np.ndarray, figure_path: str):
    # y = np.load(data_path)
    x = np.arange(y.size)
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(x, y)
    ax[0].set_title("time evolution")
    ax[1].semilogy(x, np.abs(y))
    ax[1].set_xlabel("time index")
    ax[0].set_ylabel("$\hat{u}[.]$")
    ax[1].set_ylabel("$|\hat{u}[.]|$")
    fig.savefig(figure_path, dpi=dpi)
    plt.close(fig)



def plot_state_dist(state_vectors: np.ndarray, filename: str):
    # Estimate and plot densities using matplotlib tools.
    L_1_norm = np.linalg.norm(state_vectors, ord=1, axis=0)
    L_2_norm = np.linalg.norm(state_vectors, ord=2, axis=0)
    # Similarly, compute L_infty (largest absolute value) of the analog state
    # vector.
    L_infty_norm = np.linalg.norm(state_vectors, ord=np.inf, axis=0)

    bins = 100
    plt.rcParams["figure.figsize"] = [6.40, 4.80]
    fig, ax = plt.subplots(2)
    ax[0].grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
    ax[1].grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
    # ax[0].hist(L_1_norm, bins=bins, density=True, label='$p=1$')
    ax[0].hist(L_2_norm, bins=bins, density=True, label="$p=2$")
    ax[0].hist(
        L_infty_norm, bins=bins, density=True, color="orange", label="$p=\infty$"
    )
    plt.suptitle("Estimated probability densities")
    ax[0].set_xlabel("$\|\mathbf{x}(t)\|_p$")
    ax[0].set_ylabel("$p ( \| \mathbf{x}(t) \|_p ) $")
    ax[0].legend()
    # ax[0].set_xlim((0, 1.5))
    for n in range(state_vectors.shape[0]):
        ax[1].hist(
            state_vectors[n, :],
            bins=bins,
            density=True,
            label="$x_{" + f"{n + 1}" + "}$",
        )
    ax[1].legend()
    # ax[1].set_xlim((-1, 1))
    ax[1].set_xlabel("$x(t)_n$")
    ax[1].set_ylabel("$p ( x(t)_n )$")
    fig.tight_layout()
    fig.savefig(filename)


def save_analog_frontend_parametrization(
    analog_frontend: cbadc.analog_frontend.AnalogFrontend,
    results: Dict[str, str],
    filename: str,
):
    with open(f"{filename}.txt", "w") as f:
        f.write(analog_frontend.analog_system.__str__())
        f.write(analog_frontend.digital_control.__str__())
        f.write("\nSimulation Results:\n")
        f.writelines([f"{key}: {value}\n" for key, value in results.items()])

    with open(f"{filename}.pickle", "wb") as f:
        pickle.dump(analog_frontend, f, protocol=-1)