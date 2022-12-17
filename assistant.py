import os
import numpy as np
import sounddevice as sd
import openai
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import pyttsx3


# initialize VAD block


# initialize ASR block
ASR_Checkpoint = "openai/whisper-tiny.en"
ASR_Tokenizer = WhisperProcessor.from_pretrained(ASR_Checkpoint)
ASR_Model = WhisperForConditionalGeneration.from_pretrained(ASR_Checkpoint)

# initialize LLM block
openai.api_key = os.environ["OPENAI_API_KEY"]

# initialize TTS block
engine = pyttsx3.init()

def VAD():
    flag = False
    return flag

def ASR_Transcribe(processor, model):
    freq = 16000
    duration = 5
    speech_input = sd.rec(int(duration*freq), samplerate=freq, channels=1)
    print(">Awaiting input...")
    sd.wait()

    # tokenize input
    input_features = processor(np.transpose(speech_input)[0], return_tensors="pt", sampling_rate=16000).input_features

    # generate output
    predicted_ids = model.generate(inputs=input_features, max_new_tokens=20)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def TTS(engine, txt_input):
    engine.say(txt_input)
    engine.runAndWait()
    return


while True:
    keystroke = input(">") #type d to activate ASR
    #if VAD() == False:
    if keystroke != "d":
        continue
    else:
        human_input_txt = ASR_Transcribe(ASR_Tokenizer, ASR_Model)
        print(">Human: ", human_input_txt)
        bot_reply = openai.Completion.create(model="text-davinci-003", prompt=human_input_txt, temperature=0.7, max_tokens=50)["choices"][0]["text"]
        print(">Bot Assistant: ", bot_reply[2:])
        TTS(engine, bot_reply)
