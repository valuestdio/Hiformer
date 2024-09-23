import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from vmdpy import VMD

def load_signals_from_csv(file_path):
    data = pd.read_csv(file_path)
    signals = data.values.T
    return signals

def save_vmd_results(signal, index, alpha=2000, tau=0, K=7, DC=0, init=1, tol=1e-7):
    T = len(signal)
    t = np.arange(1, T + 1) / T
    fs = 1 / T
    freqs = 2 * np.pi * (t - 0.5 - fs) / fs
    f_hat = np.fft.fftshift(np.fft.fft(signal))

    # Run VMD
    u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)

    # Create folder
    folder_name = str(index)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Save decomposed modes to CSV file
    modes_df = pd.DataFrame(u.T, columns=[f'Mode_{i+1}' for i in range(K)])
    modes_df.to_csv(os.path.join(folder_name, 'Decomposed_Modes.csv'), index=False)

    # Save plots
    linestyles = ['b', 'g', 'm', 'c', 'c', 'r', 'k']

    plt.figure()
    plt.plot(u.T)
    plt.title('Decomposed Modes')
    plt.savefig(os.path.join(folder_name, 'Decomposed_Modes.png'))

    sortIndex = np.argsort(omega[-1, :])
    omega = omega[:, sortIndex]
    u_hat = u_hat[:, sortIndex]
    u = u[sortIndex, :]

    fig1 = plt.figure(figsize=(12, 8))
    plt.subplot(K + 1, 1, 1)
    plt.plot(t, signal, 'k')
    plt.title('Original Input Signal')
    plt.xlim((0, 1))

    for k in range(K):
        plt.subplot(K + 1, 1, k + 2)
        plt.plot(t, u[k, :], linestyles[k % len(linestyles)])
        plt.title(f'Mode {k + 1}')
        plt.xlim((0, 1))

    fig1.suptitle('Original Input Signal and Decomposed Modes')
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(os.path.join(folder_name, 'Input_Signal_and_Modes.png'))

    fig2 = plt.figure()
    plt.loglog(freqs[T // 2:], abs(f_hat[T // 2:]))
    plt.xlim(np.array([1, T / 2]) * np.pi * 2)
    ax = plt.gca()
    ax.grid(which='major', axis='both', linestyle='--')
    fig2.suptitle('Input Signal Spectrum')
    plt.savefig(os.path.join(folder_name, 'Input_Signal_Spectrum.png'))

    fig3 = plt.figure()
    for k in range(K):
        plt.semilogx(2 * np.pi / fs * omega[:, k], np.arange(1, omega.shape[0] + 1), linestyles[k % len(linestyles)])
    fig3.suptitle('Center Frequency Evolution')
    plt.savefig(os.path.join(folder_name, 'Center_Frequency_Evolution.png'))

    fig4 = plt.figure()
    plt.loglog(freqs[T // 2:], abs(f_hat[T // 2:]), 'k:')
    plt.xlim(np.array([1, T / 2]) * np.pi * 2)
    for k in range(K):
        plt.loglog(freqs[T // 2:], abs(u_hat[T // 2:, k]), linestyles[k % len(linestyles)])
    fig4.suptitle('Spectral Decomposition')
    plt.savefig(os.path.join(folder_name, 'Spectral_Decomposition.png'))

    fig5 = plt.figure()
    for k in range(K):
        plt.subplot(K, 1, k + 1)
        plt.plot(t, u[k, :], linestyles[k % len(linestyles)])
        plt.xlim((0, 1))
        plt.title(f'Reconstructed Mode {k + 1}')
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(os.path.join(folder_name, 'Reconstructed_Modes.png'))

    plt.close('all')

if __name__ == '__main__':
   # Load signals from CSV file
    file_path = '-'  # Change to your CSV file path
    signals = load_signals_from_csv(file_path)

    # Perform VMD on each signal and save the results
    for i, signal in enumerate(signals):
        save_vmd_results(signal, i)
