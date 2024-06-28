from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import random
import os

app = Flask(__name__)

script_dir = os.path.dirname(__file__)
file_path_good = os.path.join(script_dir, 'Good_Food.txt')
file_path_bad = os.path.join(script_dir, 'Bad_Food.txt')

try:
    with open(file_path_good, 'r') as file:
        Goods = file.read().splitlines()
        print("File content successfully read.")
except FileNotFoundError:
    print(f"File not found: {file_path_good}")
except Exception as e:
    print(f"An error occurred: {e}")

try:
    with open(file_path_bad, 'r') as file:
        Bads = file.read().splitlines()
        print("File content successfully read.")
except FileNotFoundError:
    print(f"File not found: {file_path_bad}")
except Exception as e:
    print(f"An error occurred: {e}")

model_name = "kykim/bert-kor-base"
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = BertTokenizer.from_pretrained(model_name)

def classify_emotion(text):
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        prediction = model(**tokens)
    prediction = F.softmax(prediction.logits, dim=1)
    output = prediction.argmax(dim=1).item()
    labels = ["부정적", "긍정적"]
    emotion = labels[output]

    if emotion == "부정적":
        response = "저런... 기분이 안 좋으시군요. {} 같은 음식은 어떠신가요?".format(random.choice(Bads))
    else:
        response = "좋은 일이 있으셨군요! {}은(는) 어떠신가요?".format(random.choice(Goods))

    return emotion, response

@app.route("/")
def index_func():
    return render_template("index.html", title="Home")

@app.route("/chat")
def chat_func():
    return render_template("chat.html", title="Chat")

@app.route("/classify_emotion", methods=["POST"])
def classify_emotion_api():
    data = request.json
    text = data["text"]
    emotion, response = classify_emotion(text)
    return jsonify({"emotion": emotion, "response": response})

if __name__ == '__main__':
    app.run(debug=True)

