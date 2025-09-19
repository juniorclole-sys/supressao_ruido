"""Implementação de técnicas clássicas de supressão de ruído."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
from scipy import signal as sp_signal


@dataclass
class NoiseReductionTechnique:
    """Representa uma técnica de supressão de ruído."""

    name: str
    description: str
    function: Callable[[np.ndarray, int], np.ndarray]


def _normalize(signal: np.ndarray) -> np.ndarray:
    """Normaliza o sinal para o intervalo [-1, 1]."""

    max_abs = np.max(np.abs(signal))
    if max_abs == 0:
        return signal
    return signal / max_abs


def butter_lowpass(signal: np.ndarray, sample_rate: int, cutoff: float = 4_000.0, order: int = 6) -> np.ndarray:
    """Aplica um filtro passa-baixas de Butterworth.

    A filtragem passa-baixas é uma técnica clássica para remover ruídos de alta
    frequência que geralmente não fazem parte do sinal de interesse.
    """

    nyquist = 0.5 * sample_rate
    normalized_cutoff = cutoff / nyquist
    b, a = sp_signal.butter(order, normalized_cutoff, btype="low", analog=False)
    filtered = sp_signal.filtfilt(b, a, signal)
    return _normalize(filtered)


def moving_average(signal: np.ndarray, sample_rate: int, window_size: int = 9) -> np.ndarray:
    """Aplica um filtro de média móvel simples."""

    # Criamos um kernel uniforme e realizamos a convolução. Utilizamos o modo
    # ``same`` para preservar o tamanho do sinal de saída.
    kernel = np.ones(window_size) / window_size
    filtered = np.convolve(signal, kernel, mode="same")
    return _normalize(filtered)


def median_filter(signal: np.ndarray, sample_rate: int, kernel_size: int = 5) -> np.ndarray:
    """Aplica um filtro de mediana, eficiente para remover ruídos impulsivos."""

    filtered = sp_signal.medfilt(signal, kernel_size=kernel_size)
    return _normalize(filtered)


def wiener_filter(signal: np.ndarray, sample_rate: int, mysize: int = 29, noise: float | None = None) -> np.ndarray:
    """Aplica o filtro de Wiener utilizando a implementação da SciPy."""

    filtered = sp_signal.wiener(signal, mysize=mysize, noise=noise)
    return _normalize(filtered)


def spectral_subtraction(signal: np.ndarray, sample_rate: int, noise_frames: int = 6) -> np.ndarray:
    """Executa uma subtração espectral simples para reduzir ruídos estacionários."""

    nperseg = 1024
    noverlap = nperseg // 2

    frequencies, times, stft_matrix = sp_signal.stft(
        signal,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        window="hann",
    )

    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)

    # Estima o espectro do ruído como a média das primeiras colunas (frames) do
    # espectrograma, assumindo que elas representam apenas ruído.
    noise_estimate = magnitude[:, :noise_frames].mean(axis=1, keepdims=True)

    # Realiza a subtração espectral e evita valores negativos.
    clean_magnitude = np.maximum(magnitude - noise_estimate, 0.0)

    reconstructed = clean_magnitude * np.exp(1j * phase)

    _, processed = sp_signal.istft(
        reconstructed,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        window="hann",
    )

    # Garante que o tamanho do sinal permaneça igual ao original.
    if len(processed) > len(signal):
        processed = processed[: len(signal)]
    elif len(processed) < len(signal):
        processed = np.pad(processed, (0, len(signal) - len(processed)))

    return _normalize(np.real(processed))


TECHNIQUES: Dict[str, NoiseReductionTechnique] = {
    "Filtro Passa-Baixas": NoiseReductionTechnique(
        name="Filtro Passa-Baixas",
        description="Butterworth de 6ª ordem para remover componentes de alta frequência.",
        function=butter_lowpass,
    ),
    "Média Móvel": NoiseReductionTechnique(
        name="Média Móvel",
        description="Suavização por média móvel para reduzir ruídos de alta frequência.",
        function=moving_average,
    ),
    "Filtro de Mediana": NoiseReductionTechnique(
        name="Filtro de Mediana",
        description="Filtro não linear eficiente contra ruídos impulsivos.",
        function=median_filter,
    ),
    "Filtro de Wiener": NoiseReductionTechnique(
        name="Filtro de Wiener",
        description="Estimador ótimo no sentido de mínimos quadrados.",
        function=wiener_filter,
    ),
    "Subtração Espectral": NoiseReductionTechnique(
        name="Subtração Espectral",
        description="Técnica clássica no domínio da frequência para ruídos estacionários.",
        function=spectral_subtraction,
    ),
}


def apply_technique(name: str, signal: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, NoiseReductionTechnique]:
    """Aplica a técnica especificada ao sinal informado."""

    technique = TECHNIQUES[name]
    processed = technique.function(signal, sample_rate)
    return processed, technique


__all__ = [
    "NoiseReductionTechnique",
    "TECHNIQUES",
    "apply_technique",
    "butter_lowpass",
    "moving_average",
    "median_filter",
    "wiener_filter",
    "spectral_subtraction",
]
