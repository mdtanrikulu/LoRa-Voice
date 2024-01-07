# NCLM (Neural Codec Language Model) based voice communication over LoRa

This application enables voice communication over the LoRa protocol by converting speech to text using the [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) STT, transmitting the text data via LoRa, and then reconstructing the voice on the recipient's end with VALL-E X NCLM. It offers a practical solution for clear voice transmission over long distances using LoRa networks.

## Installation

To install the dependencies, run:

```
pip install -r requirements.txt
```

note: under `venv` meshtastic may need extra PATH configuration

## Usage

To initiate real-time transcription and translation (soon), execute the `main.py` script with the necessary arguments. For guidance on command usage and argument options, refer to the example provided within the file.

The application provides various customization options for tailoring the transcription process to your preferences. You can select the model, set the target language for translation, choose microphone settings, among other configurations. To explore and adjust these options, review and modify the arguments in the main.py script according to your requirements.
