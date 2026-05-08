📰 Fake News Detection System

An AI-powered Fake News Detection web application built using DistilBERT, Flask, and JavaScript for real-time news classification with confidence scoring.

🚀 Features
Detects whether news is Real or Fake
Uses DistilBERT Transformer Model
Real-time prediction through Flask API
Interactive frontend UI
Displays confidence score
Lightweight training optimized for low-resource systems
🧠 Tech Stack
Python
Flask
DistilBERT
PyTorch
Transformers (Hugging Face)
HTML
CSS
JavaScript
📂 Dataset

Used the ISOT Fake News Dataset containing real and fake news articles.

Dataset files:

True.csv
Fake.csv
🔁 Project Workflow
Dataset → Preprocessing → Tokenization → DistilBERT Training → Save Model → Flask API → Frontend Prediction
⚙️ Model Training

The model was trained using a lightweight fine-tuning approach:

Reduced dataset size for low memory usage
Frozen transformer layers
Trained only classification layer
Used DistilBERT tokenizer and sequence classification model
📁 Project Structure
fake-news-detector/
│
├── backend/
│   ├── app.py
│   ├── train.py
│   ├── saved_model/
│   └── test_api.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── dataset/
│   ├── True.csv
│   └── Fake.csv
│
└── README.md
▶️ How to Run
1️⃣ Clone Repository
git clone <your-repo-link>
cd fake-news-detector
2️⃣ Install Dependencies
pip install torch transformers flask flask-cors pandas scikit-learn accelerate
3️⃣ Train Model
cd backend
python train.py
4️⃣ Run Backend
python app.py

Backend runs at:

http://127.0.0.1:5000
5️⃣ Run Frontend

Open:

frontend/index.html

Or use Live Server in VS Code.

🧪 Example Input
COVID-19 was first identified in Wuhan, China in 2019.
Example Output
Real News 🟢
Confidence: 92%
🎯 Objective

The goal of this project is to automatically identify misleading or fake news articles using Natural Language Processing and Transformer-based Deep Learning techniques.

💡 Future Improvements
Deploy project online
Add user authentication
Improve accuracy with larger datasets
Add fact-checking APIs
Use advanced transformer models like RoBERTa
👨‍💻 Author

Kanchi Gupta
