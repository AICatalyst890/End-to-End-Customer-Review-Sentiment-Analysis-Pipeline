import streamlit as st
import torch
import torch.nn as nn
import time

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Sentify | Movie Review Sentiment AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Modern, Clean & Polished UX
st.markdown("""
<style>
    /* Global Styles & Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #1e1e2f 0%, #0d0e15 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin-bottom: 2rem;
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #ff7e5f, #feb47b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .hero-subtitle {
        color: #a0aec0;
        font-size: 1.05rem;
        font-weight: 300;
        margin-bottom: 0;
    }

    /* Sentiment Metric Cards */
    .result-card-pos {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: #10b981;
    }

    .result-card-neg {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: #ef4444;
    }

    /* Text Area Styling */
    .stTextArea textarea {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        background-color: #161922;
        font-size: 1rem;
        transition: border 0.3s ease;
    }

    .stTextArea textarea:focus {
        border-color: #ff7e5f;
    }

    /* Primary Action Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #ff7e5f 0%, #feb47b 100%);
        color: #0d0e15;
        font-weight: 600;
        font-size: 1.1rem;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 126, 95, 0.4);
        color: #0d0e15;
    }
</style>
""", unsafe_allow_html=True)


import joblib
import torch
import torch.nn as nn

# ==========================================
# 2. MODEL ARCHITECTURE & PREDICTION LOGIC
# ==========================================
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

@st.cache_resource
def load_artifacts():
    # 1. Load fitted TF-IDF Vectorizer
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    
    # 2. Initialize Model with vocabulary/feature size
    input_size = len(vectorizer.get_feature_names_out())
    model = RNN(input_size=input_size)
    
    # 3. Load trained weights & set to evaluation mode
    model.load_state_dict(torch.load('rnn_model.pth', map_location=torch.device('cpu')))
    model.eval()
    
    return vectorizer, model


def preprocess_and_predict(text, vectorizer, model):
    # 1. Transform raw review text using TF-IDF
    features = vectorizer.transform([text]).toarray()  # Shape: (1, num_features)
    
    # 2. Convert to PyTorch Tensor & add sequence dimension -> (1, 1, num_features)
    tensor_input = torch.tensor(features, dtype=torch.float32).unsqueeze(1)
    
    # 3. Forward Pass & Sigmoid Activation
    with torch.no_grad():
        logits = model(tensor_input)
        prob = torch.sigmoid(logits.squeeze()).item()

    return prob


# ==========================================
# 3. SIDEBAR CONTROLS & INFO
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/94/film-reel.png", width=64)
    st.title("Model Settings")
    st.caption("RNN-Based Sentiment Classifier")

    st.markdown("---")
    
    st.markdown("### ⚙️ Engine Details")
    st.markdown("""
    - **Architecture:** Recurrent Neural Network (`nn.RNN`)
    - **Hidden Layers:** 128 units
    - **Classifier:** Fully Connected + Sigmoid
    """)
    
    st.markdown("---")
    st.markdown("### 🧪 Quick Presets")
    
    if st.button("Sample Positive Review"):
        st.session_state.review_text = "This movie was an absolute masterpiece! Brilliant acting, stunning visuals, and a captivating storyline from start to finish."

    if st.button("Sample Negative Review"):
        st.session_state.review_text = "Completely uninspired and boring. The pacing was terrible and the characters felt flat and uninteresting."

    st.markdown("---")
    st.caption("Made with PyTorch & Streamlit")


# ==========================================
# 4. MAIN USER INTERFACE
# ==========================================

# Hero Header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🎬 Movie Review Sentiment AI</div>
    <div class="hero-subtitle">Instantly analyze review sentiment using PyTorch Recurrent Neural Networks</div>
</div>
""", unsafe_allow_html=True)

# Main Grid Layout
col_input, col_output = st.columns([1.2, 1], gap="large")

with col_input:
    st.subheader("📝 Input Review")
    
    default_text = st.session_state.get("review_text", "")
    review_input = st.text_area(
        label="Type or paste your review below:",
        value=default_text,
        height=220,
        placeholder="e.g. The cinematography was breathtaking, but the plot fell apart in the second half..."
    )

    analyze_btn = st.button("Analyze Sentiment ✨")

with col_output:
    st.subheader("📊 Sentiment Analysis")

    if analyze_btn:
        if not review_input.strip():
            st.warning("⚠️ Please enter a review before running analysis.")
        else:
            with st.spinner("Processing text and running inference..."):
                time.sleep(0.4)
                # Load vectorizer and trained model
                vectorizer, model = load_artifacts()
                
                # Pass input review through actual pipeline
                probability = preprocess_and_predict(review_input, vectorizer, model)
            
            # Since 1 = Positive and 0 = Negative:
            is_positive = probability >= 0.5
            confidence = probability if is_positive else (1.0 - probability)

            # Sentiment Outcome Header Card
            if is_positive:
                st.markdown(f"""
                <div class="result-card-pos">
                    <h2 style="margin:0;">😃 POSITIVE</h2>
                    <p style="margin:5px 0 0 0; font-size: 0.95rem;">Confidence score: {confidence * 100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-neg">
                    <h2 style="margin:0;">😞 NEGATIVE</h2>
                    <p style="margin:5px 0 0 0; font-size: 0.95rem;">Confidence score: {confidence * 100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            
            # Detailed Probability Gauge
            st.markdown("**Positivity Score Distribution**")
            st.progress(probability)
            
            col_pos_val, col_neg_val = st.columns(2)
            col_pos_val.metric("Positive Probability", f"{probability * 100:.1f}%")
            col_neg_val.metric("Negative Probability", f"{(1 - probability) * 100:.1f}%")

    else:
        # Default State Card
        st.info("👈 Enter a review on the left and click **Analyze Sentiment** to see prediction metrics.")