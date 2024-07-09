from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import random
import os

app = Flask(__name__)
CORS(app)  # CORS 설정 추가



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
    print("분류할 텍스트를 받았습니다:", text)  # 디버깅 출력
    try:
        emotion, response = classify_emotion(text)
        print("분류 결과:", emotion, response)  # 디버깅 출력
        return jsonify({"emotion": emotion, "response": response})
    except Exception as e:
        print(f"에러가 발생했습니다: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
