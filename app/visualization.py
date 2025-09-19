"""Rotinas de visualização para comparação dos sinais original e filtrado."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy import signal as sp_signal


def _time_axis(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """Gera o eixo de tempo correspondente ao sinal."""

    return np.linspace(0, len(signal) / sample_rate, len(signal))


def plot_waveform(ax: plt.Axes, signal: np.ndarray, sample_rate: int, title: str) -> None:
    """Desenha a forma de onda em um ``Axes`` fornecido."""

    time = _time_axis(signal, sample_rate)
    ax.plot(time, signal, color="#1f77b4", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Tempo [s]")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)


def plot_spectrogram(ax: plt.Axes, signal: np.ndarray, sample_rate: int, title: str) -> None:
    """Desenha o espectrograma utilizando escala logarítmica de potência."""

    frequencies, times, spectrogram = sp_signal.spectrogram(
        signal,
        fs=sample_rate,
        nperseg=1024,
        noverlap=512,
        scaling="density",
        mode="magnitude",
    )

    # Evita problemas numéricos ao aplicar o logaritmo.
    spectrogram_db = 20 * np.log10(spectrogram + 1e-10)

    mesh = ax.pcolormesh(times, frequencies, spectrogram_db, shading="gouraud", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Tempo [s]")
    ax.set_ylabel("Frequência [Hz]")
    ax.set_ylim(0, sample_rate / 2)

    # Adiciona uma barra de cores pequena para facilitar a leitura das potências.
    plt.colorbar(mesh, ax=ax, format="%+2.0f dB", pad=0.01)


def create_comparison_figure(
    original: np.ndarray,
    processed: np.ndarray,
    sample_rate: int,
    technique_name: str,
) -> Figure:
    """Gera uma figura contendo comparações lado a lado do sinal e do espectrograma."""

    figure, axes = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)

    plot_waveform(axes[0, 0], original, sample_rate, "Forma de onda - Entrada")
    plot_waveform(axes[0, 1], processed, sample_rate, f"Forma de onda - {technique_name}")

    plot_spectrogram(axes[1, 0], original, sample_rate, "Espectrograma - Entrada")
    plot_spectrogram(axes[1, 1], processed, sample_rate, f"Espectrograma - {technique_name}")

    figure.suptitle(f"Comparação da técnica: {technique_name}", fontsize=14)

    return figure


def save_figure(figure: Figure, output_dir: str | os.PathLike[str], technique_name: str) -> Path:
    """Salva a figura em disco e retorna o caminho gerado."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = technique_name.lower().replace(" ", "_").replace("ç", "c")
    figure_path = output_path / f"comparacao_{filename}.png"
    figure.savefig(figure_path, dpi=200)
    return figure_path


__all__ = [
    "create_comparison_figure",
    "save_figure",
    "plot_waveform",
    "plot_spectrogram",
]
