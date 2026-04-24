import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="SpamFilter AI",
    page_icon="🛡️",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

.main-badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #a78bfa;
    border: 1px solid rgba(124,106,255,0.3);
    padding: 4px 14px;
    border-radius: 100px;
    background: rgba(124,106,255,0.08);
    margin-bottom: 0.5rem;
}

.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.1;
    color: #e8e8f0;
    margin-bottom: 0.3rem;
}

.main-title span { color: #a78bfa; }

.subtitle {
    color: #6b6b88;
    font-size: 1rem;
    margin-bottom: 1.2rem;
}

.stats-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 2rem;
}

.stat-card {
    background: #13131a;
    border: 1px solid #2a2a3a;
    border-radius: 12px;
    padding: 0.75rem 1.2rem;
    flex: 1;
    min-width: 130px;
}

.stat-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6b6b88;
    margin-bottom: 4px;
}

.stat-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.25rem;
    font-weight: 700;
    color: #a78bfa;
}

.stat-sub {
    font-size: 0.72rem;
    color: #6b6b88;
    margin-top: 2px;
}

.result-spam {
    background: rgba(255,77,109,0.08);
    border: 1px solid rgba(255,77,109,0.3);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
}

.result-ham {
    background: rgba(0,214,143,0.08);
    border: 1px solid rgba(0,214,143,0.3);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
}

.verdict-spam {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #ff4d6d;
}

.verdict-ham {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #00d68f;
}

.prob-text {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    color: #6b6b88;
    margin-top: 0.3rem;
}

.word-box {
    background: #13131a;
    border: 1px solid #2a2a3a;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.95rem;
    line-height: 2;
    margin-top: 0.5rem;
}

.risky { color: #ff4d6d; font-weight: 600; background: rgba(255,77,109,0.15); padding: 2px 6px; border-radius: 4px; }
.safe  { color: #c8c8d8; }

.stTextArea textarea {
    background: #13131a !important;
    border: 1px solid #2a2a3a !important;
    color: #e8e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    border-radius: 10px !important;
}

.stTextArea textarea:focus {
    border-color: #7c6aff !important;
    box-shadow: none !important;
}

div.stButton > button {
    background: #7c6aff;
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 0.6rem 2rem;
    width: 100%;
    transition: all 0.2s;
}

div.stButton > button:hover {
    background: #a78bfa;
    transform: translateY(-1px);
}

.stProgress > div > div {
    background: linear-gradient(90deg, #ffb830, #ff4d6d) !important;
}

section[data-testid="stSidebar"] { display: none; }
header { display: none !important; }
footer { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Load & train model (cached so it only runs once) ──────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    nltk.download('stopwords', quiet=True)
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    def preprocess(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = [stemmer.stem(t) for t in text.split() if t not in stop_words]
        return ' '.join(tokens)

    # Load SMS dataset
    sms = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])
    sms['label'] = sms['label'].str.lower()
    sms_count = len(sms)

    # Load email dataset
    email = pd.read_csv('emails_trimmed.csv')
    email = email[['label', 'text']].rename(columns={'text': 'message'})
    email['label'] = email['label'].str.lower()

    # Use however many rows are actually available, balanced between spam/ham
    spam_df = email[email['label'] == 'spam']
    ham_df  = email[email['label'] == 'ham']
    n       = min(len(spam_df), len(ham_df))

    spam_sample    = spam_df.sample(n, random_state=42)
    ham_sample     = ham_df.sample(n, random_state=42)
    email_balanced = pd.concat([spam_sample, ham_sample], ignore_index=True)
    email_count    = len(email_balanced)

    df = pd.concat([sms, email_balanced], ignore_index=True).dropna(subset=['message'])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    total       = len(df)
    spam_total  = int((df['label'] == 'spam').sum())
    ham_total   = int((df['label'] == 'ham').sum())

    # Chunk preprocessing
    chunk_size = 5000
    cleaned = []
    for i in range(0, len(df), chunk_size):
        chunk = df['message'].iloc[i:i+chunk_size].apply(preprocess)
        cleaned.extend(chunk.tolist())
    df['clean'] = cleaned

    vectorizer = TfidfVectorizer(max_features=8000)
    X = vectorizer.fit_transform(df['clean'])
    y = (df['label'] == 'spam').astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = MultinomialNB()
    model.fit(X_train, y_train)

    train_count = X_train.shape[0]
    test_count  = X_test.shape[0]

    feature_names    = vectorizer.get_feature_names_out()
    top_spam_indices = model.feature_log_prob_[1].argsort()[-100:]
    top_spam_words   = set(feature_names[i] for i in top_spam_indices)

    stats = {
        "total":       total,
        "spam_total":  spam_total,
        "ham_total":   ham_total,
        "sms_count":   sms_count,
        "email_count": email_count,
        "train_count": train_count,
        "test_count":  test_count,
    }

    return model, vectorizer, stemmer, stop_words, top_spam_words, stats

# ── Header ────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Spam<span>Filter</span></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Type any message to analyse it in real time</div>', unsafe_allow_html=True)

# ── Load model with spinner ───────────────────────────────────────────
with st.spinner("Loading model... (first load takes ~60 sec)"):
    model, vectorizer, stemmer, stop_words, top_spam_words, stats = load_model()

# ── Training stats cards ──────────────────────────────────────────────
st.markdown(f"""
<div class="stats-row">
    <div class="stat-card">
        <div class="stat-label">Total Trained On</div>
        <div class="stat-value">{stats['total']:,}</div>
        <div class="stat-sub">messages</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Spam Messages</div>
        <div class="stat-value">{stats['spam_total']:,}</div>
        <div class="stat-sub">{round(stats['spam_total']/stats['total']*100, 1)}% of dataset</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Ham Messages</div>
        <div class="stat-value">{stats['ham_total']:,}</div>
        <div class="stat-sub">{round(stats['ham_total']/stats['total']*100, 1)}% of dataset</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Training Split</div>
        <div class="stat-value">{stats['train_count']:,}</div>
        <div class="stat-sub">{stats['test_count']:,} held for testing</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Sources</div>
        <div class="stat-value">2</div>
        <div class="stat-sub">SMS · Email</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Example chips ─────────────────────────────────────────────────────
examples = [
    "Win a FREE iPhone now!",
    "Hey, see you at 5pm?",
    "URGENT: Claim your cash prize",
    "Can you send the notes?",
    "You have been selected! Call now"
]

st.markdown("**Try an example:**")
cols = st.columns(len(examples))
selected_example = None
for i, ex in enumerate(examples):
    with cols[i]:
        if st.button(ex[:18] + "…" if len(ex) > 18 else ex, key=f"ex_{i}"):
            selected_example = ex

# ── Text input ────────────────────────────────────────────────────────
default_text = selected_example if selected_example else ""
message = st.text_area(
    "Message",
    value=default_text,
    placeholder="e.g. Congratulations! You've won a FREE prize. Click now to claim...",
    height=130,
    label_visibility="collapsed"
)

analyse_clicked = st.button("Analyse →")

# ── Prediction ────────────────────────────────────────────────────────
def preprocess_input(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [stemmer.stem(t) for t in text.split() if t not in stop_words]
    return ' '.join(tokens)

if analyse_clicked and message.strip():
    cleaned  = preprocess_input(message)
    vector   = vectorizer.transform([cleaned])
    prob     = float(model.predict_proba(vector)[0][1])
    label    = "SPAM" if prob > 0.5 else "HAM"
    prob_pct = round(prob * 100, 1)

    css_class     = "result-spam" if label == "SPAM" else "result-ham"
    verdict_class = "verdict-spam" if label == "SPAM" else "verdict-ham"
    icon          = "🚨" if label == "SPAM" else "✅"

    st.markdown(f"""
    <div class="{css_class}">
        <div class="{verdict_class}">{icon} {label}</div>
        <div class="prob-text">Spam probability: {prob_pct}%</div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(prob)

    # Word highlighting
    original_tokens = re.sub(r'[^a-z\s]', '', message.lower()).split()
    highlighted_html = ""
    risky_count = 0
    for word in original_tokens:
        stemmed = stemmer.stem(word)
        if stemmed in top_spam_words:
            highlighted_html += f'<span class="risky">{word}</span> '
            risky_count += 1
        else:
            highlighted_html += f'<span class="safe">{word}</span> '

    risky_label = (
        f"· {risky_count} suspicious word{'s' if risky_count != 1 else ''} flagged"
        if risky_count > 0 else "· no suspicious words"
    )
    st.markdown(f"**Word analysis** {risky_label}")
    st.markdown(f'<div class="word-box">{highlighted_html}</div>', unsafe_allow_html=True)

elif analyse_clicked and not message.strip():
    st.warning("Please enter a message first.")
