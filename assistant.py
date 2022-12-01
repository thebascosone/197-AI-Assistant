#This would be the main file na magi-initialize ng 4 models and magko-call ng VAD
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def GenerateReply(human_input):
  checkpoint = "facebook/blenderbot-400M-distill"
  tokenizer = AutoTokenizer.from_pretrained(checkpoint)
  model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

  model_inputs = tokenizer(human_input, padding=True, return_tensors='pt')
  model_replies = model.generate(**model_inputs)
  return tokenizer.batch_decode(model_replies, skip_special_tokens=True)[0]

while (1):
  txt_input = input("Human: ") #should be a string coming from the ASR block
  reply = GenerateReply([txt_input])
  print("Bot: ", reply) #instead of print, pass output to STT block
