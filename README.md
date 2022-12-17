# 197-AI-Assistant
CoE 197 Assignment 2 Group 3
 - Bryan Bascos
 - Adrian Dumosmog


### The AI Assistant would be comprised of 4 blocks: 
1. Voice Activity Detection
   - constantly listens for human speech which would activate the next block
2. Automatic Speech Recognition
   - converts human speech to text
3. Large Language Model
   - formulates an appropriate response to the query from the previous block
4. Text-To-Speech
   - converts the response to audible form


### Command stuff:
```
git clone https://github.com/thebascosone/197-AI-Assistant.git
cd 197-AI-Assistant
pip3 install -r requirements.txt
python3 assistant.py
```

### Accessing the LLM from OpenAI:
The code right now is configured such that it connects to a flask server that we made and access openAI's LLM model from there to avoid issues with sharing API keys. To bypass this step you may copy the get_llm_response function from backendserver.py and paste it into assistant.py and insert a valid openai api key.