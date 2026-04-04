import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="AI Review Detector",
    page_icon="🕵️",
    layout="centered"
)

# ===================== NLTK SETUP =====================
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# ===================== LOAD MODEL =====================
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open('model/model.pkl', 'rb'))
        vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))
        return model, vectorizer
    except FileNotFoundError:
        return None, None

model, vectorizer = load_model()

if model is None or vectorizer is None:
    st.error("Model files not found. Please run train.py first.")
    st.stop()

# ===================== TEXT CLEANING =====================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# ===================== RULE BASED CHECK =====================
def is_obviously_fake(text):
    exclamations = text.count('!')
    words = text.split()
    caps_words = [w for w in words if w.isupper() and len(w) > 2]
    if exclamations >= 4:
        return True
    if len(caps_words) >= 3:
        return True
    return False

# ===================== STYLING =====================
st.markdown("""
<style>
/* Remove toolbar and gap */
.block-container { padding-top: 2rem !important; }
header { visibility: hidden; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }

/* Background — clean white */
.stApp { background-color: #f8f9fc; }

/* Header */
.header-wrap {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    margin-bottom: 4px;
}
.main-title {
    font-size: 2.3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #6c63ff, #48cfad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.subtitle {
    color: #6b7280;
    text-align: center;
    font-size: 0.92rem;
    margin-bottom: 6px;
}

/* Stats */
.stats-container {
    display: flex;
    justify-content: center;
    gap: 16px;
    margin: 18px 0;
}
.stat-box {
    background: #ffffff;
    border-radius: 12px;
    padding: 14px 20px;
    text-align: center;
    border: 1px solid #e5e7eb;
    flex: 1;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.stat-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #6c63ff;
}
.stat-label {
    font-size: 0.7rem;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Input label */
.input-label {
    color: #374151;
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 6px;
}

/* Text area */
.stTextArea textarea {
    background-color: #ffffff !important;
    border: 1.5px solid #e5e7eb !important;
    border-radius: 12px !important;
    color: #1f2937 !important;
    font-size: 0.95rem !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
}
.stTextArea textarea::placeholder {
    color: #9ca3af !important;
    opacity: 1 !important;
}
.stTextArea textarea:focus {
    border-color: #6c63ff !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(90deg, #6c63ff, #48cfad) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 30px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    width: 100% !important;
    box-shadow: 0 4px 15px rgba(108,99,255,0.3) !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* Result cards */
.result-fake {
    background: #fff5f5;
    border: 1.5px solid #ff416c;
    border-radius: 15px;
    padding: 28px;
    text-align: center;
    margin-top: 16px;
    box-shadow: 0 4px 15px rgba(255,65,108,0.1);
}
.result-genuine {
    background: #f0fdf9;
    border: 1.5px solid #48cfad;
    border-radius: 15px;
    padding: 28px;
    text-align: center;
    margin-top: 16px;
    box-shadow: 0 4px 15px rgba(72,207,173,0.1);
}
.result-title {
    font-size: 1.7rem;
    font-weight: 800;
    margin-bottom: 4px;
}
.result-subtitle {
    font-size: 0.88rem;
    color: #6b7280;
}
.confidence-score {
    font-size: 2rem;
    font-weight: 800;
    margin-top: 12px;
}

/* Review stats */
.review-stats {
    display: flex;
    justify-content: center;
    gap: 16px;
    margin-top: 16px;
}
.review-stat {
    background: #ffffff;
    border-radius: 10px;
    padding: 10px 20px;
    text-align: center;
    border: 1px solid #e5e7eb;
    flex: 1;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
.review-stat-value {
    font-size: 1.2rem;
    font-weight: 700;
    color: #1f2937;
}
.review-stat-label {
    font-size: 0.7rem;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Disclaimer box */
.disclaimer {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-radius: 10px;
    padding: 12px 16px;
    color: #92400e;
    font-size: 0.82rem;
    margin-top: 16px;
    line-height: 1.5;
}

/* How it works */
.how-it-works {
    background: #ffffff;
    border-radius: 15px;
    padding: 20px 28px;
    border: 1px solid #e5e7eb;
    margin-top: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.step {
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 12px 0;
    color: #6b7280;
    font-size: 0.88rem;
}
.step-number {
    background: linear-gradient(135deg, #6c63ff, #48cfad);
    color: white;
    border-radius: 50%;
    width: 26px;
    height: 26px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.78rem;
    flex-shrink: 0;
}

/* Divider */
hr { border-color: #e5e7eb !important; }
</style>
""", unsafe_allow_html=True)

# ===================== HEADER =====================
st.markdown("""
<div class="header-wrap">
    <span style="font-size:2.6rem;">🕵️</span>
    <span class="main-title">AI Review Detector</span>
</div>
<div class="subtitle">Detect AI-generated fake reviews using Machine Learning & NLP</div>
""", unsafe_allow_html=True)

# ===================== STATS =====================
st.markdown("""
<div class="stats-container">
    <div class="stat-box">
        <div class="stat-value">88%</div>
        <div class="stat-label">Accuracy</div>
    </div>
    <div class="stat-box">
        <div class="stat-value">40K+</div>
        <div class="stat-label">Reviews Trained</div>
    </div>
    <div class="stat-box">
        <div class="stat-value">NLP</div>
        <div class="stat-label">Technology</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ===================== INPUT =====================
st.markdown('<p class="input-label">📋 Paste a product review below:</p>', unsafe_allow_html=True)
review_text = st.text_area(
    label="",
    height=170,
    placeholder="e.g. This phone has amazing battery life and the camera quality is excellent..."
)

detect = st.button("Analyze Review", use_container_width=True)

# ===================== PREDICTION =====================
if detect:
    if not review_text.strip():
        st.warning("Please enter a review before clicking Analyze.")
    else:
        st.markdown("---")
        word_count = len(review_text.split())
        char_count = len(review_text)

        if is_obviously_fake(review_text):
            st.markdown(f"""
            <div class="result-fake">
                <div class="result-title" style="color:#ff416c;">FAKE REVIEW</div>
                <div class="result-subtitle">Detected by rule-based filter — excessive caps or punctuation</div>
                <div class="confidence-score" style="color:#ff416c;">99%</div>
                <div class="result-subtitle">Confidence Score</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(0.99)

        else:
            cleaned = clean_text(review_text)
            if not cleaned.strip():
                st.warning("The review became empty after processing. Please enter a more meaningful review.")
            else:
                vectorized = vectorizer.transform([cleaned])
                prediction = model.predict(vectorized)[0]
                probabilities = model.predict_proba(vectorized)[0]
                score = round(max(probabilities) * 100, 2)

                if prediction == 'CG':
                    st.markdown(f"""
                    <div class="result-fake">
                        <div class="result-title" style="color:#ff416c;">FAKE REVIEW</div>
                        <div class="result-subtitle">This review appears to be AI-generated</div>
                        <div class="confidence-score" style="color:#ff416c;">{score}%</div>
                        <div class="result-subtitle">Confidence Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-genuine">
                        <div class="result-title" style="color:#48cfad;">GENUINE REVIEW</div>
                        <div class="result-subtitle">This review appears to be written by a real human</div>
                        <div class="confidence-score" style="color:#48cfad;">{score}%</div>
                        <div class="result-subtitle">Confidence Score</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.progress(score / 100)

        # Review stats
        st.markdown(f"""
        <div class="review-stats">
            <div class="review-stat">
                <div class="review-stat-value">{word_count}</div>
                <div class="review-stat-label">Words</div>
            </div>
            <div class="review-stat">
                <div class="review-stat-value">{char_count}</div>
                <div class="review-stat-label">Characters</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Disclaimer
        st.markdown("""
        <div class="disclaimer">
            ⚠️ <strong>Important:</strong> This prediction is based on patterns learned from a specific dataset
            and may not always be accurate. The model is trained to detect subtle AI-generated reviews
            that match the dataset's patterns — it may misclassify reviews that look different from
            its training data. Results should not be treated as definitive.
        </div><br>
        """, unsafe_allow_html=True)

        with st.expander("View Processed Review Text"):
            st.write(clean_text(review_text))

# ===================== HOW IT WORKS =====================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("#### ⚙️ How it works:")
st.markdown("""
<div class="how-it-works">
    <div class="step">
        <div class="step-number">1</div>
        <span>A rule-based filter first checks for obvious spam signals like excessive caps or punctuation</span>
    </div>
    <div class="step">
        <div class="step-number">2</div>
        <span>If not obvious, text is cleaned — punctuation and common words are removed</span>
    </div>
    <div class="step">
        <div class="step-number">3</div>
        <span>Cleaned text is converted into numbers using TF-IDF vectorization</span>
    </div>
    <div class="step">
        <div class="step-number">4</div>
        <span>A Logistic Regression model trained on 40,000+ reviews predicts the result</span>
    </div>
    <div class="step">
        <div class="step-number">5</div>
        <span>A confidence score shows how certain the model is about its prediction</span>
    </div>
</div>
""", unsafe_allow_html=True)