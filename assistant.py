import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import soundfile as sf
import torch
torch.set_num_threads(1)

from IPython.display import Audio
from pprint import pprint

import io
import numpy as np
import torch
torch.set_num_threads(1)
import torchaudio
import matplotlib
import matplotlib.pylab as plt
torchaudio.set_audio_backend("soundfile")
import pyaudio

import os
import openai

#from jupyterplot import ProgressPlot
import threading

import json
import queue


import argparse
import tempfile
import queue
import sys

import time

from gtts import gTTS
import os
from flask import request, Response

import requests
q = queue.Queue

#torch.hub.download_url_to_file('https://models.silero.ai/vad_models/en.wav', 'en_example.wav')

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)

(get_speech_timestamps,
 _, read_audio,
 *_) = utils

sampling_rate = 16000 # also accepts 8000


FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = int(SAMPLE_RATE / 10)

audio = pyaudio.PyAudio()


num_samples = 1536

def validate(model,
             inputs: torch.Tensor):
    with torch.no_grad():
        outs = model(inputs)
    return outs

# Provided by Alexander Veysov
def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/abs_max
    sound = sound.squeeze()  # depends on the use case
    return sound

def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

def voice_assistant():
    
    stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    data = []
    voiced_confidences = []
    
    global continue_recording
    continue_recording = True
    
    #pp = ProgressPlot(plot_names=["Silero VAD"],line_names=["speech probabilities"], x_label="audio chunks")
    
    stop_listener = threading.Thread()
    stop_listener.start()
    global count
    cooldown = 0
    count = 0

    global message_length 
    message_length = 0
    print("Assistant is now on.")
    while True:
    
        audio_chunk = stream.read(num_samples)
    
        # in case you want to save the audio later
        data.append(audio_chunk)
    
        audio_int16 = np.frombuffer(audio_chunk, np.int16);

        audio_float32 = int2float(audio_int16)
    
        # get the confidences and add them to the list to plot them later
        new_confidence = model(torch.from_numpy(audio_float32), 16000).item()
        voiced_confidences.append(new_confidence)
        
        if cooldown > 0:
            new_confidence = 0
            cooldown-=1

        if(new_confidence > .90 and count < 1):
            cooldown = 5
            print("Voice detected. Starting ASR Subroutine...")
            count+=1
            AI_response = threading.Thread(target=ASR_subroutine)
            AI_response.start()
            
            print("New Thread Made")
            
            AI_response.join()
            print("Thread done")
            time.sleep((message_length/10))
            new_confidence = 0
            print("Listening again...")
        
            
        #print(new_confidence) 
        #pp.update(new_confidence)

    print(voiced_confidences)
    #pp.finalize()

def ASR_subroutine():
    freq = 44100

    duration = 5

    recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
    sd.wait()
    write("recording0.wav", freq, recording)
    model = whisper.load_model("tiny.en")

    result = model.transcribe("recording0.wav")
    print("Finished recording...")
    print(result["text"])

    get_LLM_response(result["text"])


def get_LLM_response(prompt):
    message = requests.post("http://192.168.1.2:5000/openai", data=prompt)
    print(message.text)
    
    message = message.text
    global count

    global message_length
    message_length = len(message)
    count -=1
    TTS_response(message)

def TTS_response(message):
    
    audio = gTTS(text=message, lang="en", slow=False)
    audio.save("TTS.mp3")
    os.system("TTS.mp3")


voice_assistant()