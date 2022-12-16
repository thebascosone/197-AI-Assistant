import sounddevice as sd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

def ASR_Transcribe(voice_input):
    # load model and tokenizer
    checkpoint = "openai/whisper-tiny.en"
    processor = WhisperProcessor.from_pretrained(checkpoint)
    model = WhisperForConditionalGeneration.from_pretrained(checkpoint)

    # tokenize
    input_features = processor(voice_input, return_tensors="pt", sampling_rate=16000).input_features

    # retrieve logits
    #logits = model(input_features, decoder_input_ids = torch.tensor([[50258]])).logits

    # take argmax and decode
    #predicted_ids = torch.argmax(logits, dim=-1)
    generated_ids = model.generate(inputs=input_features, max_new_tokens=20)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def GenerateReply(human_input):
    # load model and tokenizer
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # tokenize input and generate output
    user_input_ids = tokenizer.encode(human_input + tokenizer.eos_token, return_tensors='pt')
    #bot_input_ids = torch.cat([chat_history_ids, user_input_ids], dim=-1) if step > 0 else user_input_ids
    bot_input_ids = user_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # decode batch ids into human language
    return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

freq = 16000
duration = 5

#get sound input
stream = sd.rec(int(duration*freq), samplerate=freq, channels=1)
print("Speak, mortal!")
sd.wait()

#speech recognition block
human_input_txt = ASR_Transcribe(np.transpose(stream)[0])
print("Human: ", human_input_txt)

#large language model block
bot_reply = GenerateReply(human_input_txt)
print("ChatBot: ", bot_reply)
