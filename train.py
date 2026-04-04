# IMPORTS
import pandas as pd
import pickle
import re
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# LOAD DATASET 
print("Loading dataset...")
df = pd.read_csv('data/fake reviews dataset.csv')

# KEEP REQUIRED COLUMNS 
df = df[['text_', 'label']]

# REMOVE EMPTY ROWS 
df = df.dropna()

# Remove rows with empty text after stripping spaces
df = df[df['text_'].str.strip() != ""]

# CHECK LABELS 
print("\nUnique labels in dataset:")
print(df['label'].unique())

print("\nLabel distribution:")
print(df['label'].value_counts())

# TEXT CLEANING FUNCTION 
def clean_text(text):
    text = str(text).lower()
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# APPLY CLEANING
df['cleaned'] = df['text_'].apply(clean_text)

# FEATURES & LABELS 
X = df['cleaned']
y = df['label']

# TRAIN / TEST SPLIT 
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# TF-IDF VECTORIZATION 
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"\nTF-IDF train shape: {X_train_tfidf.shape}")
print(f"TF-IDF test shape: {X_test_tfidf.shape}")

# TRAIN MODEL
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_tfidf, y_train)

# EVALUATE MODEL 
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\n===================== MODEL EVALUATION =====================")
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nModel classes:")
print(model.classes_)

# CREATE MODEL FOLDER IF NOT EXISTS 
os.makedirs('model', exist_ok=True)

# SAVE MODEL & VECTORIZER 
with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Optional: save class labels separately for clarity
with open('model/classes.pkl', 'wb') as f:
    pickle.dump(model.classes_, f)

print("\n✅ Model, vectorizer, and classes saved successfully!")