import pandas as pd
import re 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
# Load the datasets
fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")
fake['label'] = 0
real['label'] = 1
# Combine the datasets
df = pd.concat([fake, real], axis=0)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
# Preprocess the text data
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)     # remove URLs
    text = re.sub(r"\d+", "", text)         # remove numbers
    text = re.sub(r"[^\w\s]", "", text)     # remove punctuation
    return text.strip()

# Combine title + text and clean
df['content'] = (df['title'] + " " + df['text']).apply(clean_text)
#spliting the data into features and labels
X = df['content']
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000,stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_vec) 
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
# Save the model and vectorizer
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

with open("model/evaluation.txt", "w") as f:
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
    f.write(classification_report(y_test, y_pred))
