# Step 1: Import libraries
import pandas as pd
import numpy as np
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if not already
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Step 2: Load dataset
df = pd.read_csv(r"C:\Users\isaad\Desktop\Sentiment Analysis of Movie Reviews\IMDB Dataset.csv\IMDB Dataset.csv")  # Download from Kaggle
print(df.head())

# Step 3: Text Preprocessing function
def preprocess_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'<.*?>', '', text)  # remove HTML tags
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    words = text.split()
    words = [word for word in words if word not in stop_words]  # remove stopwords
    return " ".join(words)

df['review'] = df['review'].apply(preprocess_text)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42
)

# Step 5: TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 6: Model Training (Logistic Regression)
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Step 7: Predictions
y_pred = model.predict(X_test_tfidf)

# Step 8: Evaluation
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, pos_label='positive')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
