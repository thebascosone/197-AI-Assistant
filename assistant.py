import numpy as np
import sounddevice as sd
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import pyttsx3

openai.api_key = "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"

# initialize VAD block

# initialize ASR block
ASR_Checkpoint = "openai/whisper-tiny.en"
ASR_Tokenizer = WhisperProcessor.from_pretrained(ASR_Checkpoint)
ASR_Model = WhisperForConditionalGeneration.from_pretrained(ASR_Checkpoint)

# initialize LLM block
LLM_Checkpoint = "microsoft/DialoGPT-medium"
LLM_Tokenizer = AutoTokenizer.from_pretrained(LLM_Checkpoint)
LLM_Model = AutoModelForCausalLM.from_pretrained(LLM_Checkpoint)

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

    # tokenize
    input_features = processor(np.transpose(speech_input)[0], return_tensors="pt", sampling_rate=16000).input_features

    # retrieve logits
    #logits = model(input_features, decoder_input_ids = torch.tensor([[50258]])).logits

    # take argmax and decode
    #predicted_ids = torch.argmax(logits, dim=-1)
    predicted_ids = model.generate(inputs=input_features, max_new_tokens=20)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def GenerateReply(human_input, tokenizer, model):
    # tokenize input
    user_input_ids = tokenizer.encode(human_input + tokenizer.eos_token, return_tensors='pt')
    #bot_input_ids = torch.cat([chat_history_ids, user_input_ids], dim=-1) if step > 0 else user_input_ids
    bot_input_ids = user_input_ids

    # generate output
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    #decode batch ids into human language
    return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

def TTS(engine, txt_input):
    engine.say(txt_input)
    engine.runAndWait()
    return


while (1):
    keystroke = input(">") #type d to activate ASR
    #if VAD() == False:
    if keystroke != "d":
        continue
    else:
        human_input_txt = ASR_Transcribe(ASR_Tokenizer, ASR_Model)
        print(">Human: ", human_input_txt)
        bot_reply = openai.Completion.create(model="text-davinci-003", prompt=human_input_txt, temperature=0.7, max_tokens=50)["choices"][0]["text"]
        #bot_reply = GenerateReply(human_input_txt, LLM_Tokenizer, LLM_Model)
        print(">Bot Assistant: ", bot_reply[2:])
        TTS(engine, bot_reply)
