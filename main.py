import argparse
import io
import speech_recognition as sr
import nltk
from nltk.tokenize import sent_tokenize

# meshtastic
import meshtastic
import meshtastic.serial_interface
import meshtastic.tcp_interface
from pubsub import pub

# vall-e
from vall_e.utils.prompt_making import make_prompt
from vall_e.utils.generation import SAMPLE_RATE, generate_audio, preload_models
from IPython.display import Audio



from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform
from faster_whisper import WhisperModel

def setup_clone_voice():
    make_prompt(name="paimon", audio_prompt_path="./models/voice.wav")


def onReceive(packet, interface): # called when a packet arrives
    Decoded  = packet.get('decoded')
    Message  = Decoded.get('text')
    To       = packet.get('to')
    From     = packet.get('from')

    if(Message):
        print('Received: From:', From, '-', Message)
        audio_array = generate_audio(Message)
        Audio(audio_array, rate=SAMPLE_RATE)

def main():


    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Selects the model size for processing. Available options: 'tiny', 'base', 'small', 'medium', 'large'.",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--device", default="cpu", help="Specifies the computing device for CTranslate2 inference. Options: 'auto', 'cuda', 'cpu'.",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--compute_type", default="int8", help="Determines the quantization type for computation. Choices include 'auto', 'int8', 'int8_float16', 'float16', 'int16', 'float32'.",
                        choices=["auto", "int8", "int8_floatt16", "float16", "int16", "float32"])
    parser.add_argument("--translation_lang", default='English',
                        help="Sets the target language for translation.", type=str)
    parser.add_argument("--non_english", action='store_true',
                        help="Enables the use of a non-English model.")
    parser.add_argument("--threads", default=8,
                        help="Defines the number of threads for CPU-based inference.", type=int)
    parser.add_argument("--energy_threshold", default=2000,
                        help="Sets the microphone's energy level threshold for detection.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="Determines the delay in seconds for real-time recording.", type=float)

    parser.add_argument("--phrase_timeout", default=3,
                        help="Specifies the duration in seconds of inactivity before considering it as the end of a phrase in transcription.", type=float)


    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                    help="Specifies the default microphone for SpeechRecognition. Use 'list' to see available microphones.", type=str)
    args = parser.parse_args()

    # Timestamp for when the last recording was retrieved from the queue.
    phrase_time = None
    # Buffer for storing the raw audio bytes.
    last_sample = bytes()
    # A thread-safe Queue to transfer data from the audio recording callback function.
    data_queue = Queue()
    # SpeechRecognizer is utilized for recording audio due to its ability to detect the end of spoken phrases.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Important to set this. Disabling dynamic energy threshold prevents the recognizer from continuously recording by lowering the energy threshold excessively.
    recorder.dynamic_energy_threshold = False

    # Initializes a serial interface for Meshtastic device communication.
    interface = meshtastic.serial_interface.SerialInterface()

    # Subscribes to the 'meshtastic.receive' event to handle incoming data with the 'onReceive' function.
    pub.subscribe(onReceive, "meshtastic.receive")

    # Sets up the environment for cloning voices, initializing necessary configurations.
    setup_clone_voice()

    # Loads the required models into memory in advance for faster processing.
    preload_models()

    # Crucial for Linux users to avoid application hangs and crashes due to selecting an incorrect microphone.
    if 'linux' in platform:
        mic_name = args.default_microphone
        # Lists available microphones if 'list' is passed or no microphone name is provided.
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            # Selects the specified microphone by matching the provided name.
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break

    else:
        source = sr.Microphone(sample_rate=16000)

    if args.model == "large":
        args.model = "large-v2"

    model = args.model
    if args.model != "large-v2" and not args.non_english:
        model = model + ".en"

    # todo implement local (offline) translation with this flag
    translation_lang = args.translation_lang

    device = args.device
    if device == "cpu":
        compute_type = "int8"
    else:
        compute_type = args.compute_type
    cpu_threads = args.threads

    nltk.download('punkt')
    audio_model = WhisperModel(model, device = device, compute_type = compute_type , cpu_threads = cpu_threads)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    temp_file = NamedTemporaryFile().name
    transcription = ['']

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        """
        A threaded callback function that receives audio data at the end of a recording.
        :param audio: An AudioData object containing the recorded audio bytes.
        """
        # Extract raw audio data and enqueue it into a thread-safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Initiating a background thread to receive raw audio bytes.
    # SpeechRecognizer's helper function is used for ease of implementation.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Inform the user that the model is ready for use.
    print("Model loaded.\n")

    while True:
        try:
            now = datetime.utcnow()
            # Retrieve raw audio data from the queue if available.
            if not data_queue.empty():
                phrase_complete = False
                # Determine if a sufficient gap has occurred between recordings to signify the end of a phrase.
                # Reset the audio buffer for new data if the phrase is complete.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True
                # Update the timestamp of the latest audio data reception.
                phrase_time = now

                # Append new audio data to the existing buffer.
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                # Convert the raw audio buffer to WAV format.
                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Write the WAV data to a temporary file.
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                # Transcribe the audio to text.
                text = ""
                segments, info = audio_model.transcribe(temp_file)
                for segment in segments:
                    text += segment.text

                # Update the transcription list based on whether a new phrase has started.
                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text
                last_four_elements = transcription[-10:]
                result = ''.join(last_four_elements)
                sentences = sent_tokenize(result)
                last_sentence = sentences.pop()
                print("message:", last_sentence)
                # Send the Meshtastic message here.
                interface.sendText(last_sentence, wantAck=True)

                # Introduce a short pause to avoid overloading the processor in the loop.
                sleep(0.25)
        except KeyboardInterrupt:
            # Gracefully close the interface upon a keyboard interrupt and exit the loop.
            interface.close()
            break


    print("\n\nTranscription:")
    for line in transcription:
        print(line)


if __name__ == "__main__":
    main()
