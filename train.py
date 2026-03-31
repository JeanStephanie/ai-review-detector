# IMPORTS 
import pandas as pd
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# DOWNLOAD NLTK DATA 
nltk.download('stopwords')

# LOAD DATASET 
df = pd.read_csv('data/fake reviews dataset.csv')

# KEEP ONLY WHAT WE NEED
df = df[['text_', 'label']]

# REMOVE EMPTY ROWS 
df = df.dropna()

# TEXT CLEANING FUNCTION 
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalpha() or char == ' '])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# APPLY CLEANING 
df['cleaned'] = df['text_'].apply(clean_text)

# SPLIT FEATURES AND LABEL
X = df['cleaned']
y = df['label']

# SPLIT INTO TRAIN AND TEST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CONVERT TEXT TO NUMBERS
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# TRAIN THE MODEL
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# EVALUATE THE MODEL 
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nDetailed Report:\n", classification_report(y_test, y_pred))

# SAVE THE MODEL AND VECTORIZER 
pickle.dump(model, open('model/model.pkl', 'wb'))
pickle.dump(vectorizer, open('model/vectorizer.pkl', 'wb'))

print("\nModel saved successfully!")