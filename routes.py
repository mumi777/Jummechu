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
        print("Good_Food.txt 파일을 성공적으로 읽었습니다.")
except FileNotFoundError:
    print(f"파일을 찾을 수 없습니다: {file_path_good}")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")

try:
    with open(file_path_bad, 'r') as file:
        Bads = file.read().splitlines()
        print("Bad_Food.txt 파일을 성공적으로 읽었습니다.")
except FileNotFoundError:
    print(f"파일을 찾을 수 없습니다: {file_path_bad}")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")

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
    print("분류할 텍스트를 받았습니다:", text) # 디버깅 출력
    try:
        emotion, response = classify_emotion(text)
        print("분류 결과:", emotion, response) # 디버깅 출력
        return jsonify({"emotion": emotion, "response": response})
    except Exception as e:
        print(f"에러가 발생했습니다: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

