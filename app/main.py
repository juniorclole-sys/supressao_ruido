"""Interface gráfica da aplicação de supressão de ruído."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from . import audio_utils
from .processing import TECHNIQUES, NoiseReductionTechnique, apply_technique
from .visualization import create_comparison_figure, save_figure


class NoiseReductionApp:
    """Janela principal responsável pela interação com o usuário."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Aplicação de Supressão de Ruído")
        self.root.geometry("1200x800")

        # Armazena o sinal carregado e o último resultado processado.
        self.original_signal: Optional[np.ndarray] = None
        self.sample_rate: Optional[int] = None
        self.last_results: Dict[str, np.ndarray] = {}

        # Variáveis de estado exibidas na interface.
        self.selected_file = tk.StringVar(value="Nenhum arquivo selecionado.")
        self.status_message = tk.StringVar(value="Selecione ou grave um áudio para começar.")
        self.record_duration = tk.DoubleVar(value=5.0)
        self.current_technique: Optional[NoiseReductionTechnique] = None
        self.current_canvas: Optional[FigureCanvasTkAgg] = None

        # Diretório padrão onde as figuras são salvas.
        self.figures_dir = "output/figuras"

        self._build_layout()

    # ------------------------------------------------------------------
    # Construção da interface
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        """Cria os widgets que compõem a janela principal."""

        self._build_file_section()
        self._build_record_section()
        self._build_technique_section()
        self._build_actions_section()
        self._build_status_bar()
        self._build_plot_area()

    def _build_file_section(self) -> None:
        """Cria a área responsável pelo carregamento de arquivos."""

        frame = ttk.LabelFrame(self.root, text="Seleção de arquivo", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(frame, text="Selecionar arquivo", command=self._select_file).pack(side=tk.LEFT)
        ttk.Label(frame, textvariable=self.selected_file, width=80).pack(side=tk.LEFT, padx=10)

    def _build_record_section(self) -> None:
        """Cria a área responsável pela gravação de áudio."""

        frame = ttk.LabelFrame(self.root, text="Gravação", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(frame, text="Duração (s):").pack(side=tk.LEFT)
        ttk.Entry(frame, textvariable=self.record_duration, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="Gravar áudio", command=self._record_audio).pack(side=tk.LEFT)

    def _build_technique_section(self) -> None:
        """Cria a lista de técnicas disponíveis para processamento."""

        frame = ttk.LabelFrame(self.root, text="Técnicas clássicas", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)

        self.technique_list = tk.Listbox(frame, height=5)
        self.technique_list.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Preenche a lista com os nomes das técnicas definidas em ``processing.py``.
        for name in TECHNIQUES:
            self.technique_list.insert(tk.END, name)

        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.technique_list.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.technique_list.config(yscrollcommand=scrollbar.set)

    def _build_actions_section(self) -> None:
        """Cria os botões que disparam as operações principais."""

        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(frame, text="Aplicar técnica selecionada", command=self._process_selected).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(frame, text="Gerar imagens de todas", command=self._process_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="Salvar áudio processado", command=self._save_last_audio).pack(side=tk.LEFT, padx=5)

    def _build_status_bar(self) -> None:
        """Cria uma barra na parte inferior da janela para mensagens de status."""

        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(frame, textvariable=self.status_message).pack(side=tk.LEFT)

    def _build_plot_area(self) -> None:
        """Cria a região utilizada para exibir as figuras."""

        frame = ttk.LabelFrame(self.root, text="Comparação gráfico/espectrograma", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.plot_frame = frame

    # ------------------------------------------------------------------
    # Ações dos botões
    # ------------------------------------------------------------------
    def _select_file(self) -> None:
        """Permite ao usuário escolher um arquivo de áudio via diálogo padrão."""

        file_path = filedialog.askopenfilename(
            title="Selecione um arquivo de áudio",
            filetypes=[("Arquivos WAV", "*.wav"), ("Arquivos FLAC", "*.flac"), ("Todos", "*.*")],
        )

        if not file_path:
            return

        try:
            signal, sample_rate = audio_utils.load_audio(file_path)
        except Exception as exc:  # noqa: BLE001 - informamos qualquer erro ao usuário.
            messagebox.showerror("Erro ao carregar", f"Não foi possível carregar o arquivo: {exc}")
            return

        self.original_signal = signal
        self.sample_rate = sample_rate
        self.selected_file.set(file_path)
        self.status_message.set("Arquivo carregado com sucesso. Selecione uma técnica para processar.")

    def _record_audio(self) -> None:
        """Aciona a gravação de áudio utilizando o microfone padrão."""

        try:
            duration = float(self.record_duration.get())
        except ValueError:
            messagebox.showerror("Valor inválido", "Informe uma duração numérica para a gravação.")
            return

        if duration <= 0:
            messagebox.showerror("Valor inválido", "A duração deve ser positiva.")
            return

        try:
            signal, sample_rate, file_path = audio_utils.record_audio(duration)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Erro na gravação", f"Não foi possível gravar: {exc}")
            return

        self.original_signal = signal
        self.sample_rate = sample_rate
        self.selected_file.set(str(file_path))
        self.status_message.set("Gravação concluída. Selecione uma técnica para processar.")

    def _process_selected(self) -> None:
        """Processa o áudio com a técnica destacada na lista."""

        if self.original_signal is None or self.sample_rate is None:
            messagebox.showwarning("Áudio não carregado", "Carregue ou grave um áudio antes de processar.")
            return

        selection = self.technique_list.curselection()
        if not selection:
            messagebox.showwarning("Seleção vazia", "Escolha uma técnica na lista.")
            return

        technique_name = self.technique_list.get(selection[0])
        self._apply_and_display(technique_name)

    def _process_all(self) -> None:
        """Aplica todas as técnicas e gera as figuras correspondentes no disco."""

        if self.original_signal is None or self.sample_rate is None:
            messagebox.showwarning("Áudio não carregado", "Carregue ou grave um áudio antes de processar.")
            return

        saved_paths = []
        for technique_name in TECHNIQUES:
            processed, technique = apply_technique(technique_name, self.original_signal, self.sample_rate)
            self.last_results[technique_name] = processed

            figure = create_comparison_figure(self.original_signal, processed, self.sample_rate, technique.name)
            path = save_figure(figure, self.figures_dir, technique.name)
            saved_paths.append(path)
            plt.close(figure)

        self.status_message.set("Figuras geradas para todas as técnicas.")
        messagebox.showinfo("Concluído", f"Figuras salvas em: {self.figures_dir}\n\n" + "\n".join(map(str, saved_paths)))

    def _save_last_audio(self) -> None:
        """Permite salvar o último áudio processado em disco."""

        if not self.current_technique or self.current_technique.name not in self.last_results:
            messagebox.showwarning(
                "Nada para salvar", "Primeiro aplique alguma técnica para gerar um áudio processado."
            )
            return

        processed = self.last_results[self.current_technique.name]
        file_path = filedialog.asksaveasfilename(
            title="Salvar áudio processado",
            defaultextension=".wav",
            filetypes=[("WAV", "*.wav")],
            initialfile=f"processado_{self.current_technique.name.replace(' ', '_')}.wav",
        )

        if not file_path:
            return

        try:
            audio_utils.save_audio(file_path, processed, self.sample_rate or 44100)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Erro ao salvar", f"Não foi possível salvar o arquivo: {exc}")
            return

        self.status_message.set(f"Áudio salvo em: {file_path}")

    # ------------------------------------------------------------------
    # Rotinas auxiliares
    # ------------------------------------------------------------------
    def _apply_and_display(self, technique_name: str) -> None:
        """Executa a técnica especificada e exibe o resultado na interface."""

        processed, technique = apply_technique(technique_name, self.original_signal, self.sample_rate)
        self.last_results[technique.name] = processed
        self.current_technique = technique

        figure = create_comparison_figure(self.original_signal, processed, self.sample_rate, technique.name)
        self._show_figure(figure)
        self.status_message.set(f"Técnica '{technique.name}' aplicada com sucesso.")

        # Após renderizar no Tkinter, salvamos automaticamente a figura.
        saved_path = save_figure(figure, self.figures_dir, technique.name)
        self.status_message.set(
            f"Técnica '{technique.name}' aplicada. Figura salva em: {saved_path}"
        )

    def _show_figure(self, figure) -> None:
        """Renderiza a figura no ``Frame`` destinado à visualização."""

        if self.current_canvas is not None:
            self.current_canvas.get_tk_widget().destroy()

        canvas = FigureCanvasTkAgg(figure, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.current_canvas = canvas

    # ------------------------------------------------------------------
    # Execução da aplicação
    # ------------------------------------------------------------------


def main() -> None:
    """Função de inicialização da interface."""

    root = tk.Tk()
    NoiseReductionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
