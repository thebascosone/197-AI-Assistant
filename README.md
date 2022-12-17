# 197-AI-Assistant: FLeWm
CoE 197 Assignment 2 Group 3
 - Bryan Bascos
 - Adrian Dumosmog
 - @Launama
 - @Mr.Leeds


### The AI Assistant is comprised of 4 blocks: 
1. Voice Activity Detection
   - constantly listens for human speech which would activate the next block
2. Automatic Speech Recognition
   - converts human speech to text
3. Large Language Model
   - formulates an appropriate response to the query from the previous block
4. Text-To-Speech
   - converts the response to audible form


### Prerequisite:
- As this Assistant uses an OpenAI LLM, you must have an OpenAI account to generate an API Key.
- Replace \<yourkey\> in the code below (keep the quotes) with your own API Key.


### Command line:
```
setx OPENAI_API_KEY “<yourkey>”
git clone https://github.com/thebascosone/197-AI-Assistant.git
cd 197-AI-Assistant
pip3 install -r requirements.txt
python3 assistant.py
```
