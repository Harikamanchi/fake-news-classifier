import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Dataset load
# Kaggle lo "Fake News Dataset" untundi. Sample CSV lo columns: [title, text, label]
df = pd.read_csv("data/news.csv")

# 2. Data split
x_train, x_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# 3. Text vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = vectorizer.fit_transform(x_train)
tfidf_test = vectorizer.transform(x_test)

# 4. Model train
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train, y_train)

# 5. Predictions
y_pred = model.predict(tfidf_test)

# 6. Accuracy
score = accuracy_score(y_test, y_pred)
print(f"Accuracy: {round(score*100, 2)}%")

# 7. Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
