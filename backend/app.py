from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

app = Flask(__name__)
CORS(app)

print("🚀 Starting app...")

model = None
tokenizer = None

try:
    model = DistilBertForSequenceClassification.from_pretrained(
        "./saved_model",
        use_safetensors=True
    )
    tokenizer = DistilBertTokenizerFast.from_pretrained("./saved_model")

    model.eval()
    print("✅ Model loaded")

except Exception as e:
    print("❌ Model load error:", e)


def predict(text):
    if model is None or tokenizer is None:
        raise Exception("Model not loaded properly")

    if not text or text.strip() == "":
        return 0, 0.0

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    pred = torch.argmax(probs).item()
    conf = float(torch.max(probs))

    return pred, conf


@app.route("/")
def home():
    return "API Running ✅"


@app.route("/predict", methods=["POST"])
def predict_news():
    try:
        data = request.get_json()

        # 🔒 Safe validation
        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data["text"]
        print("📩 Input:", text)

        pred, conf = predict(text)

        return jsonify({
            "prediction": "Real News 🟢" if pred == 1 else "Fake News 🔴",
            "confidence": f"{conf*100:.2f}%"
        })

    except Exception as e:
        print("❌ API ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ✅ Run server
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
