"""Utilidades para carregar, gravar e salvar arquivos de áudio."""

from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf


# Diretório padrão onde os áudios gravados serão armazenados.
RECORDINGS_DIR = Path("recordings")
RECORDINGS_DIR.mkdir(exist_ok=True)


def load_audio(file_path: str | os.PathLike[str]) -> Tuple[np.ndarray, int]:
    """Carrega um arquivo de áudio e retorna o sinal em mono e a taxa de amostragem.

    Parameters
    ----------
    file_path:
        Caminho para o arquivo de áudio que será carregado.

    Returns
    -------
    sinal, taxa_de_amostragem
        O sinal retornado é convertido para ponto flutuante e normalizado para o
        intervalo ``[-1, 1]`` para facilitar o processamento pelas técnicas de
        supressão.
    """

    # O soundfile lê o áudio já em floating point, porém pode retorná-lo em
    # formato estéreo. Convertemos para ``float32`` para economizar memória.
    signal, sample_rate = sf.read(file_path, dtype="float32")

    # Caso o arquivo seja estéreo (Nx2), fazemos a média dos canais para obter
    # um sinal mono. Isso simplifica o processamento e garante que todas as
    # técnicas tratem os dados de forma uniforme.
    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    # Normalizamos para o intervalo [-1, 1] para evitar saturação em etapas
    # posteriores de processamento.
    max_abs = np.max(np.abs(signal))
    if max_abs > 0:
        signal = signal / max_abs

    return signal.astype(np.float32), int(sample_rate)


def save_audio(
    file_path: str | os.PathLike[str], signal: np.ndarray, sample_rate: int
) -> None:
    """Salva um sinal de áudio no disco utilizando o formato WAV.

    Parameters
    ----------
    file_path:
        Caminho onde o arquivo será gravado.
    signal:
        Sinal que será persistido. Caso esteja fora do intervalo ``[-1, 1]`` ele
        é automaticamente normalizado.
    sample_rate:
        Taxa de amostragem do arquivo a ser gravado.
    """

    # Garante que o diretório de destino exista antes da escrita.
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    # Ajusta o sinal para evitar clipes durante a gravação.
    max_abs = np.max(np.abs(signal))
    if max_abs > 1.0:
        normalized = signal / max_abs
    else:
        normalized = signal

    sf.write(file_path, normalized.astype(np.float32), sample_rate)


def record_audio(
    duration: float,
    sample_rate: int = 44_100,
    channels: int = 1,
) -> Tuple[np.ndarray, int, Path]:
    """Grava um áudio a partir do microfone padrão utilizando ``sounddevice``.

    Parameters
    ----------
    duration:
        Duração da gravação, em segundos.
    sample_rate:
        Taxa de amostragem desejada. Por padrão utilizamos 44.1 kHz.
    channels:
        Número de canais (1 para mono, 2 para estéreo). A aplicação trabalha com
        mono, mas deixamos o parâmetro configurável para maior flexibilidade.

    Returns
    -------
    sinal_gravado, taxa_de_amostragem, caminho_arquivo
        O sinal é retornado em ``float32`` e também salvo automaticamente no
        diretório ``recordings`` com um nome baseado na data e hora da captura.
    """

    # Calcula o número total de amostras da gravação a partir da duração.
    num_samples = int(duration * sample_rate)

    # Informa ao usuário (pela saída padrão) que a captura foi iniciada.
    print(f"Iniciando gravação por {duration:.1f} segundos...")

    # ``sounddevice.rec`` retorna um array NxC (amostras x canais). Definimos o
    # dtype para float32 para ser compatível com o restante da pipeline.
    recording = sd.rec(num_samples, samplerate=sample_rate, channels=channels, dtype="float32")

    # A chamada é assíncrona, por isso precisamos aguardar o término.
    sd.wait()

    print("Gravação concluída.")

    # Caso a gravação tenha múltiplos canais, convertemos para mono realizando a
    # média dos canais.
    if channels > 1:
        recording_mono = recording.mean(axis=1)
    else:
        recording_mono = recording.reshape(-1)

    # Define o nome padrão do arquivo utilizando o timestamp atual.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = RECORDINGS_DIR / f"gravacao_{timestamp}.wav"

    # Persiste o áudio gravado no disco para que o usuário possa reutilizar.
    save_audio(file_path, recording_mono, sample_rate)

    return recording_mono, sample_rate, file_path


__all__ = ["load_audio", "save_audio", "record_audio", "RECORDINGS_DIR"]
