# Aplicativo de Supressão de Ruído

Este projeto implementa uma aplicação gráfica em Python para estudo e comparação de técnicas clássicas de supressão de ruído aplicadas a sinais de áudio. É possível carregar um arquivo existente ou gravar um novo áudio, aplicar filtros e visualizar comparações lado a lado das formas de onda e espectrogramas antes e depois do processamento.

## Funcionalidades

- Carregamento de arquivos WAV/FLAC.
- Gravação de áudio utilizando o microfone padrão (biblioteca `sounddevice`).
- Aplicação das técnicas clássicas:
  - Filtro passa-baixas de Butterworth.
  - Filtro de média móvel.
  - Filtro de mediana.
  - Filtro de Wiener.
  - Subtração espectral simples.
- Geração automática de figuras comparativas (forma de onda e espectrograma) salvas em `output/figuras`.
- Opção para salvar o áudio processado.

## Configuração do ambiente

1. Crie e ative um ambiente virtual (opcional, mas recomendado).
2. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

   > **Importante:** a biblioteca `sounddevice` depende de drivers específicos do sistema. Em ambientes Linux pode ser necessário instalar `portaudio`.

## Execução

Para iniciar a aplicação execute:

```bash
python -m app.main
```

A janela exibirá controles para selecionar/gravar áudio, escolher uma técnica e visualizar os resultados em tempo real.

As figuras geradas automaticamente ficam disponíveis em `output/figuras` e o áudio filtrado pode ser salvo manualmente através do botão dedicado.

## Estrutura do código

- `app/audio_utils.py`: utilidades para leitura, gravação e escrita de arquivos de áudio.
- `app/processing.py`: implementação das técnicas clássicas de supressão de ruído.
- `app/visualization.py`: rotinas de geração de gráficos e espectrogramas.
- `app/main.py`: interface gráfica construída com Tkinter.

Todos os módulos estão amplamente comentados, descrevendo o propósito de cada função e etapa do processamento.
