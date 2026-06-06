"""
Aplikasi Web Deteksi Phishing Berbasis AI
Menggunakan Streamlit dan Model Machine Learning
"""

import streamlit as st
import pickle
import pandas as pd
import os

# ==========================================
# KONFIGURASI HALAMAN STREAMLIT
# ==========================================

st.set_page_config(
    page_title="PhishGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Professional Dark Theme dengan Blue Accents
st.markdown("""
    <style>
    /* Root colors */
    :root {
        --primary-blue: #00d4ff;
        --secondary-blue: #0096ff;
        --dark-bg: #0a1128;
        --card-bg: #1a2847;
        --success-green: #00ff88;
        --danger-red: #ff3366;
    }
    
    /* Main background */
    .main {
        background-color: #0a1128;
        color: #ffffff;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #0f1a2e;
        padding: 2rem 1rem;
    }
    
    .sidebar .sidebar-content h1 {
        color: #00d4ff;
        font-size: 1.5em;
        margin-bottom: 0.5rem;
    }
    
    /* Header Title */
    .title-section {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #0096ff 0%, #00d4ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Input Field */
    .stTextInput input {
        background-color: #1a2847 !important;
        border: 2px solid #0096ff !important;
        border-radius: 8px !important;
        color: #ffffff !important;
        padding: 12px !important;
        font-size: 1rem !important;
    }
    
    .stTextInput input::placeholder {
        color: #00d4ff !important;
        opacity: 0.7 !important;
    }
    
    /* Button */
    .stButton button {
        background: linear-gradient(135deg, #0096ff 0%, #00d4ff 100%);
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 14px 30px !important;
        font-size: 1.1rem !important;
        font-weight: bold !important;
        width: 100% !important;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #00d4ff 0%, #0096ff 100%);
    }
    
    /* Card styling */
    .result-card {
        background-color: #1a2847;
        border: 1px solid #0096ff;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
    }
    
    .result-card-success {
        background-color: rgba(0, 255, 136, 0.1);
        border: 2px solid #00ff88;
    }
    
    .result-card-danger {
        background-color: rgba(255, 51, 102, 0.1);
        border: 2px solid #ff3366;
    }
    
    /* Status Text */
    .status-aman {
        color: #00ff88;
        font-weight: bold;
        font-size: 1.5em;
    }
    
    .status-phishing {
        color: #ff3366;
        font-weight: bold;
        font-size: 1.5em;
    }
    
    /* Metric boxes */
    .stMetric {
        background-color: #1a2847;
        border: 1px solid #0096ff;
        border-radius: 8px;
        padding: 1.5rem !important;
    }
    
    .stMetric label {
        color: #00d4ff;
        font-size: 0.9em;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1a2847;
        border: 1px solid #0096ff;
        border-radius: 8px;
        color: #00d4ff;
    }
    
    /* Info box */
    .info-box {
        background-color: #1a2847;
        border-left: 4px solid #0096ff;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# FUNCTION: LOAD MODEL
# ==========================================

@st.cache_resource
def load_model(model_path='model_phishing.pkl'):
    """
    Load model dari file pickle (dengan caching untuk performa)
    """
    if not os.path.exists(model_path):
        st.error(f"❌ File '{model_path}' tidak ditemukan!")
        st.info("💡 Silakan jalankan 'train_model.py' terlebih dahulu untuk membuat model.")
        st.stop()
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


# ==========================================
# FUNCTION: EXTRACT FEATURES FROM URL
# ==========================================

def extract_features(url):
    """
    Ekstraksi fitur dari URL:
    - panjang_url: jumlah karakter dalam URL
    - ada_karakter_at: ada tidaknya simbol '@' (0 atau 1)
    - jumlah_titik: jumlah karakter '.' dalam URL
    """
    
    panjang_url = len(url)
    ada_karakter_at = 1 if '@' in url else 0
    jumlah_titik = url.count('.')
    
    return {
        'panjang_url': panjang_url,
        'ada_karakter_at': ada_karakter_at,
        'jumlah_titik': jumlah_titik
    }


# ==========================================
# FUNCTION: PREDICT URL
# ==========================================

def predict_url(url, model):
    """
    Prediksi apakah URL aman atau phishing
    Return: (prediction, confidence)
    """
    features = extract_features(url)
    
    # Buat DataFrame untuk input model
    feature_df = pd.DataFrame([features])
    
    # Prediksi
    prediction = model.predict(feature_df)[0]
    confidence = max(model.predict_proba(feature_df)[0]) * 100
    
    return prediction, confidence, features


# ==========================================
# MAIN APP INTERFACE
# ==========================================

# Header dengan Custom HTML
st.title("🛡️ Deteksi Website Phishing Berbasis AI")
st.markdown("---")

# Subtitle & Deskripsi
st.markdown("""
    Aplikasi ini menggunakan **Machine Learning (Random Forest)** untuk mendeteksi 
    apakah sebuah URL website adalah **Aman** atau **Phishing** berdasarkan analisis fitur URL.
    
    **Fitur yang dianalisis:**
    - 📏 Panjang URL
    - 🔗 Ada/tidaknya karakter '@' dalam URL
    - 📍 Jumlah titik ('.') dalam URL
""")

st.markdown("---")

# Load model
model = load_model('model_phishing.pkl')

# Input section - ANALISIS URL
st.markdown("---")
st.markdown("### 🔍 Analisis URL")

col1, col2 = st.columns([4, 1])

with col1:
    url_input = st.text_input(
        label="Masukkan URL website:",
        placeholder="https://example.com",
        label_visibility="collapsed"
    )

with col2:
    analyze_button = st.button("🚀 Analisis", use_container_width=True)

# Proses analisis
if analyze_button:
    if not url_input.strip():
        st.warning("⚠️ Silakan masukkan URL terlebih dahulu!")
    else:
        # Prediksi
        prediction, confidence, features = predict_url(url_input, model)
        
        st.markdown("---")
        st.markdown("### 📊 Hasil Analisis")
        
        # Layout hasil
        result_col1, result_col2 = st.columns([2, 1])
        
        with result_col1:
            if prediction == 0:  # Aman
                st.markdown("""
                <div class="result-card result-card-success">
                <div style="text-align: center;">
                <div style="font-size: 3em;">✓</div>
                <div class="status-aman">AMAN</div>
                <div style="font-size: 1.2em; margin-top: 0.5rem;">Confidence Score</div>
                <div style="font-size: 2.5em; color: #00ff88;">{:.2f}%</div>
                <div style="background-color: #00ff88; height: 4px; margin-top: 1rem; border-radius: 2px;"></div>
                </div>
                </div>
                """.format(confidence), unsafe_allow_html=True)
                
                st.success("""
                Website ini terdeteksi sebagai **AMAN** untuk dikunjungi. 
                Namun, selalu berhati-hati dan perhatikan URL dengan seksama sebelum membuka link.
                """)
            else:  # Phishing
                st.markdown("""
                <div class="result-card result-card-danger">
                <div style="text-align: center;">
                <div style="font-size: 3em;">⚠️</div>
                <div class="status-phishing">PHISHING</div>
                <div style="font-size: 1.2em; margin-top: 0.5rem;">Confidence Score</div>
                <div style="font-size: 2.5em; color: #ff3366;">{:.2f}%</div>
                <div style="background-color: #ff3366; height: 4px; margin-top: 1rem; border-radius: 2px;"></div>
                </div>
                </div>
                """.format(confidence), unsafe_allow_html=True)
                
                st.error("""
                ⛔ **WEBSITE PHISHING - BERBAHAYA!**
                
                **JANGAN MEMBUKA LINK INI!** Ini adalah upaya penipuan untuk:
                - Mencuri informasi pribadi
                - Mengakses akun bank/email
                - Memasang malware
                
                Laporkan ke support layanan terkait.
                """)
        
        with result_col2:
            # Stats
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background-color: #1a2847; border-radius: 8px; border: 1px solid #0096ff;">
            <div style="margin: 1rem 0;">
            <div style="color: #00d4ff; font-size: 0.8em;">Status</div>
            <div style="color: #ffffff; font-size: 1.2em; margin-top: 0.5rem;">
            {} {}
            </div>
            </div>
            
            <div style="margin: 1rem 0;">
            <div style="color: #00d4ff; font-size: 0.8em;">Confidence</div>
            <div style="color: #ffffff; font-size: 1.2em; margin-top: 0.5rem;">
            {:.2f}%
            </div>
            </div>
            
            <div style="margin: 1rem 0;">
            <div style="color: #00d4ff; font-size: 0.8em;">Model</div>
            <div style="color: #ffffff; font-size: 1.2em; margin-top: 0.5rem;">
            Random Forest
            </div>
            </div>
            
            <div style="margin: 1rem 0;">
            <div style="color: #00d4ff; font-size: 0.8em;">Akurasi Model</div>
            <div style="color: #ffffff; font-size: 1.2em; margin-top: 0.5rem;">
            ~95%
            </div>
            </div>
            </div>
            """.format("✅ AMAN" if prediction == 0 else "⚠️ PHISHING", "✓" if prediction == 0 else "✗", confidence), unsafe_allow_html=True)
        
        # Detail fitur
        with st.expander("📋 Detail Fitur yang Dianalisis"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📏 Panjang URL", f"{features['panjang_url']} karakter")
            with col2:
                status_at = "✓ Ada" if features['ada_karakter_at'] == 1 else "✗ Tidak"
                st.metric("🔗 Karakter '@'", status_at)
            with col3:
                st.metric("📍 Jumlah Titik", f"{features['jumlah_titik']} titik")

# Sidebar
with st.sidebar:
    st.markdown("### 🛡️ PHISHGUARD AI")
    st.markdown("*AI-Powered Phishing Detection*")
    st.markdown("---")
    
    st.markdown("### ℹ️ Informasi")
    with st.container():
        st.markdown("""
        **TENTANG APLIKASI**
        - 🤖 Model: Random Forest Classifier
        - 📊 Dataset: 500 sampel URL
        - 📈 Akurasi: ~95%
        - 🔍 Fitur: 3 fitur basis URL
        """)
    
    st.markdown("---")
    st.markdown("### 📖 CARA MENGGUNAKAN")
    st.markdown("""
    1. ✍️ Masukkan URL di kolom input
    2. 🔎 Klik tombol "Analisis URL"
    3. 👀 Lihat hasil prediksi
    4. ⚠️ Baca penjelasan detail
    """)
    
    st.markdown("---")
    st.markdown("### 🔐 TIPS KEAMANAN")
    st.markdown("""
    - 🔗 Periksa URL sebelum klik
    - 🔒 Gunakan HTTPS
    - 🛡️ Aktifkan 2FA
    - ⚠️ Hindari email mencurigakan
    """)
    
    st.markdown("---")
    st.markdown("### 💻 DEVELOPER INFO")
    st.markdown("""
    - Framework: **Streamlit**
    - ML: **scikit-learn**
    - Language: **Python 3.8+**
    """)

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 2.5em; background: linear-gradient(135deg, #00d4ff 0%, #0096ff 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        Deteksi Website Phishing Berbasis AI
        </h1>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Aplikasi ini menggunakan <b>Machine Learning (Random Forest)</b> untuk mendeteksi apakah sebuah URL website adalah <b>✅ Aman</b> atau <b>⚠️ Phishing</b> berdasarkan analisis fitur URL.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem;">
    <span style="background-color: #00ff88; color: #0a1128; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold;">
    🟢 AI SECURITY
    </span>
    </div>
    """, unsafe_allow_html=True)
