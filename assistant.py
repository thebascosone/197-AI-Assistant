import os
import torch
import numpy as np
import threading
import pyaudio
import sounddevice as sd
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import openai
import pyttsx3

# initialize VAD block
VAD_Model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)

(get_speech_timestamps,
 _, read_audio,
 *_) = utils

# initialize ASR block
ASR_Checkpoint = "openai/whisper-tiny.en"
ASR_Tokenizer = WhisperProcessor.from_pretrained(ASR_Checkpoint)
ASR_Model = WhisperForConditionalGeneration.from_pretrained(ASR_Checkpoint)

# initialize LLM block
openai.api_key = os.environ["OPENAI_API_KEY"]

# initialize TTS block
engine = pyttsx3.init()

def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/abs_max
    sound = sound.squeeze()  # depends on the use case
    return sound

def ASR_Transcribe(processor, model):
    freq = 16000
    duration = 5
    speech_input = sd.rec(int(duration*freq), samplerate=freq, channels=1)
    sd.wait()

    # tokenize input
    input_features = processor(np.transpose(speech_input)[0], return_tensors="pt", sampling_rate=16000).input_features

    # generate output
    predicted_ids = model.generate(inputs=input_features, max_new_tokens=20)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def GenerateReply(human_input_txt):
    global count
    global message_length
    reply = openai.Completion.create(model="text-davinci-003", prompt=human_input_txt, temperature=0.7, max_tokens=140)
    message_length = len(reply["choices"][0]["text"])
    count-=1
    return reply["choices"][0]["text"]

def TTS(engine, txt_input):
    engine.say(txt_input)
    engine.runAndWait()
    return


FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = int(SAMPLE_RATE / 10)

audio = pyaudio.PyAudio()

num_samples = 1536

# start the audio input stream
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

global continue_recording
global count
global message_length 

# initialize Assistant
voiced_confidences = []
continue_recording = True
Listener = threading.Thread()
Listener.start()
cooldown = 0
count = 0
message_length = 0
print("FLeWm Assistant is online!")


while (1):
    print(">Awaiting input...")

    # get input
    audio_chunk = stream.read(num_samples)
    audio_int16 = np.frombuffer(audio_chunk, np.int16)
    audio_float32 = int2float(audio_int16)

    # get the confidences
    new_confidence = VAD_Model(torch.from_numpy(audio_float32), 16000).item()
    voiced_confidences.append(new_confidence)

    if cooldown > 0: # no human speech detected
        new_confidence = 0
        cooldown-=1

    if(new_confidence > .90 and count < 1): # human speech detected
        cooldown = 5
        print(">!!!")
        count+=1

        human_input_txt = ASR_Transcribe(ASR_Tokenizer, ASR_Model)
        print(">Human: ", human_input_txt)
        bot_reply = GenerateReply(human_input_txt)
        print(">FLeWm: ", bot_reply[1:])
        TTS(engine, bot_reply)

        new_confidence = 0
