#This would be the main file na magi-initialize ng 4 models and magko-call ng VAD
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

checkpoint = "facebook/blenderbot-400M-distill"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

txt_input = input("Human: ") #should be a string coming from the ASR
inputs = tokenizer([txt_input], padding=True, return_tensors='pt')
replies = model.generate(**inputs)
print("Bot: ", tokenizer.batch_decode(replies, skip_special_tokens=True)[0]) #instead of print, pass output to STT block as string
