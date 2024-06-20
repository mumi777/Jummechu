from flask import Flask, render_template, url_for, request
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
        Goods = file.read()
        print("File content successfully read.")
except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")

try:
    with open(file_path_bad, 'r') as file:
        Bads = file.read()
        print("File content successfully read.")
except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")

list_Good = [Goods]
list_Bad = [Bads]


model_name = "kykim/bert-kor-base"
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = BertTokenizer.from_pretrained(model_name)

def classify_emotion(text):
    # 텍스트 토큰화, 패딩
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    # 예측
    with torch.no_grad():
        prediction = model(**tokens)

		# 감정 예측 및 출력
    prediction = F.softmax(prediction.logits, dim=1)
    output = prediction.argmax(dim=1).item()
    labels = ["부정적", "긍정적"]
    goose = labels[output]
    if goose == "부정적":
        ask = "저런... 기분이 안좋으시군요. {} 같은 음식은 어떠신가요?".format(random.choice(list_Bad))

    elif goose == "긍정적":
        ask = "좋은 일이 있으셨군요! {}은(는) 어떠신가요?".format(random.choice(list_Good))
    return render_template(
        "chat.html", title="Chat", 
    )
@app.route("/")
def index_func():
    return render_template("index.html", title="Home")

@app.route("/chat", methods=["GET", "POST"])
def chat_func():
    data = request.get_json()
    sentence = data["text"]
    result = classify_emotion(sentence)
    return render_template("chat.html", title="Chat")


if __name__ == '__main__':
	app.run(debug=True)
