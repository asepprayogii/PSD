import streamlit as st
import numpy as np
import librosa
import os
import pickle
import joblib

# =====================================================
# 🎨 Styling & Tampilan Streamlit
# =====================================================
st.set_page_config(page_title="🎧 Audio Buka/Tutup Classifier", layout="centered")

st.markdown("""
    <style>
    body {
        background-color: #0E1117;
        color: #E0E0E0;
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp {
        background-color: #1E1E1E;
    }
    .title {
        text-align: center;
        color: #00C9A7;
        font-size: 2.2em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .result-box {
        background-color: #111;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0px 0px 10px #00C9A7;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🎧 Prediksi Audio — Buka atau Tutup</div>', unsafe_allow_html=True)
st.caption("Unggah file audio (.wav) dan sistem akan menebak apakah kondisi **Buka** atau **Tutup** berdasarkan model Random Forest.")

# =====================================================
# 📦 Fungsi untuk Memuat Model
# =====================================================
@st.cache_resource
def load_model():
    model_path = "random_forest_model.pkl"
    if not os.path.exists(model_path):
        st.error("❌ File model tidak ditemukan: random_forest_model.pkl")
        st.stop()

    # Coba load model
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception:
        model = joblib.load(model_path)
        return model

model = load_model()

# =====================================================
# 📤 Upload File Audio
# =====================================================
uploaded_file = st.file_uploader("📁 Pilih file audio (.wav atau .mp3)", type=["wav", "mp3"])

# =====================================================
# 🧩 Proses dan Prediksi Audio
# =====================================================
def extract_features(path):
    y, sr = librosa.load(path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_mean = np.mean(mfcc_delta.T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))
    features = np.hstack([mfcc_mean, mfcc_delta_mean, zcr, rms])
    return features

if uploaded_file is not None:
    # Simpan file sementara
    temp_audio = "temp_audio.wav"
    with open(temp_audio, "wb") as f:
        f.write(uploaded_file.read())

    # Tampilkan audio player
    st.audio(temp_audio)
    st.info("🎵 Putar audio di atas untuk memastikan file sudah benar.")

    try:
        # Ekstraksi fitur
        features = extract_features(temp_audio)
        st.write(f"📊 Total fitur yang diekstraksi: {len(features)}")

        # Tombol prediksi
        if st.button("🔮 Prediksi Audio"):
            pred = model.predict([features])[0]

            # Interpretasi hasil (ubah angka ke label)
            if pred == 1 or str(pred).lower() == "buka":
                hasil = "🔊 **Buka**"
                color = "#00FFAA"
            else:
                hasil = "🔕 **Tutup**"
                color = "#FF5555"

            st.markdown(
                f"""
                <div class="result-box" style="border:2px solid {color}; color:{color}">
                    <h3>Hasil Prediksi:</h3>
                    <h1>{hasil}</h1>
                </div>
                """, unsafe_allow_html=True
            )

    except Exception as e:
        st.error(f"❌ Gagal memproses audio: {e}")

else:
    st.info("Silakan unggah file audio terlebih dahulu untuk memulai prediksi.")

# =====================================================
# 📘 Panduan
# =====================================================
with st.expander("📘 Petunjuk Penggunaan"):
    st.markdown("""
    1. Pastikan file `random_forest_model.pkl` ada di folder yang sama dengan `app-audio.py`.
    2. Unggah file `.wav` (disarankan).
    3. Klik tombol **Prediksi Audio** untuk melihat hasil.
    4. Hasil akan menampilkan status **Buka** atau **Tutup**.
    """)

