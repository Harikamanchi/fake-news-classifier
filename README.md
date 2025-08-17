# fake-news-classifier
A machine learning project to classify Fake vs Real news.
# 📰 Fake News Classifier

This project uses **Machine Learning (ML)** to classify news articles as **Fake** or **Real**.

---

## 📌 Project Overview
- Fake news detection is a major challenge in today's digital world.  
- This project builds a **text classification model** using Python and scikit-learn.  
- It uses a **TF-IDF Vectorizer** for feature extraction and a **Passive Aggressive Classifier** for classification.  

---

## ⚙️ Tech Stack
- Python 🐍  
- Pandas  
- Scikit-learn  
- NLTK  

---

## 📂 Project Structurefake-news-classifier/
│
├── dataset/               # Dataset CSV file (train & test)
├── fake_news_classifier.py # Main Python script
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

## How to run
git clone https://github.com/Harikamanchi/fake-news-classifier.git
cd fake-news-classifier

# Install Dependencies
pip install -r requirements.txt

# Run the project
python fake_news_classifier.py

# Sample Output
Accuracy: 92.5%
Confusion Matrix:
[[589  40]
 [ 45 598]]


