from flask import Flask, Response, request
import openai

app = Flask(__name__)

@app.route("/openai", methods=['POST'])
def get_LLM_response():
    print("Received from client: {}".format(request.data))
    print(request.data.decode('utf-8'))
    openai.api_key = ("insert openai api key here")
    response = openai.Completion.create(
        model="text-curie-001",
        prompt=request.data.decode('utf-8'),
        max_tokens=200,
        temperature=0
    )

    message = response["choices"][0]["text"]
    global count
    print(message)
    global message_length
    message_length = len(message)
    #count -=1
    return message
    


@app.route('/')
def hello():
    return 'Hello World'

if __name__ == '__main__':
    app.run(debug=True, host='192.168.1.2', port=5000)