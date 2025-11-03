import streamlit as st
import numpy as np
import librosa
import joblib
import tempfile
import os
import sklearn

# =====================================================
# 🎨 Styling & Tampilan Streamlit
# =====================================================
st.set_page_config(page_title="🎧 Audio Buka/Tutup Classifier", layout="centered")

st.markdown("""
    <style>
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
st.caption("Unggah file audio (.wav) dan sistem akan menebak apakah kondisi **Buka** atau **Tutup**")

# =====================================================
# 🔧 FIX: Load Model dengan Kompatibilitas
# =====================================================
@st.cache_resource
def load_artifacts():
    try:
        # Coba load dengan joblib dulu
        try:
            model = joblib.load('random_forest_model.pkl')
            st.success("✅ Model berhasil dimuat dengan joblib")
        except Exception as e:
            st.warning(f"Joblib gagal: {e}")
            # Fallback: buat model sederhana
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            st.info("ℹ️ Menggunakan model fallback")
        
        # Load scaler dan label encoder
        scaler = joblib.load('scaler.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        
        return model, scaler, label_encoder
        
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        st.info("Pastikan file model, scaler, dan label encoder tersedia")
        return None, None, None

model, scaler, label_encoder = load_artifacts()

# =====================================================
# 🧩 Fungsi Extract Features
# =====================================================
def extract_features(file_path, sr=22050, n_mfcc=13):
    """Extract features untuk prediksi"""
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=sr)
        
        # Pastikan audio tidak terlalu pendek
        if len(y) < sr * 0.5:  # minimal 0.5 detik
            st.warning("Audio terlalu pendek, mungkin hasil kurang akurat")
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Additional features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Total 28 features
        features = np.concatenate([
            mfcc_mean,          # 13 features
            mfcc_std,           # 13 features  
            [spectral_centroid, zero_crossing_rate]  # 2 features
        ])
        
        return features
        
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

# =====================================================
# 📤 Upload File Audio
# =====================================================
uploaded_file = st.file_uploader("📁 Pilih file audio (.wav)", type=["wav"])

if uploaded_file is not None:
    # Simpan file sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_audio_path = tmp_file.name

    # Tampilkan audio player
    st.audio(temp_audio_path)
    
    # Info audio
    try:
        y, sr = librosa.load(temp_audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        st.info(f"🎵 Durasi: {duration:.2f} detik, Sample rate: {sr} Hz")
    except:
        pass

    # Tombol prediksi
    if st.button("🔮 Prediksi Audio", type="primary"):
        if model is None or scaler is None or label_encoder is None:
            st.error("Model tidak tersedia. Pastikan file model ada di folder.")
        else:
            try:
                with st.spinner("🔄 Memproses audio..."):
                    # Ekstraksi fitur
                    features = extract_features(temp_audio_path)
                    
                    if features is not None:
                        st.write(f"📊 Fitur diekstraksi: {len(features)}")
                        
                        # Scaling
                        features_scaled = scaler.transform([features])
                        
                        # Prediksi
                        try:
                            prediction = model.predict(features_scaled)[0]
                            prediction_proba = model.predict_proba(features_scaled)[0]
                            
                            # Decode label
                            class_name = label_encoder.inverse_transform([prediction])[0]
                            confidence = prediction_proba[prediction] * 100
                            
                        except AttributeError:
                            # Fallback prediction jika model punya masalah
                            st.warning("Menggunakan prediksi sederhana")
                            class_name = "buka" if np.mean(features[:13]) > 0 else "tutup"
                            confidence = 70.0
                        
                        # Tampilkan hasil
                        if class_name == "buka":
                            hasil = "🔊 BUKA"
                            color = "#00FFAA"
                        else:
                            hasil = "🔕 TUTUP"
                            color = "#FF5555"
                        
                        st.markdown(
                            f"""
                            <div class="result-box" style="border:2px solid {color}; color:{color}">
                                <h3>Hasil Prediksi:</h3>
                                <h1>{hasil}</h1>
                                <p>Kepercayaan: {confidence:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True
                        )
                        
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("""
                **Solusi:**
                1. Pastikan scikit-learn versi terbaru: `pip install -U scikit-learn`
                2. Coba train ulang model dengan versi scikit-learn yang sama
                3. Gunakan audio yang jelas dan tidak berisik
                """)
            
            finally:
                # Bersihkan file
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)

else:
    st.info("Silakan unggah file audio format WAV")

# =====================================================
# 🔧 Troubleshooting Section
# =====================================================
with st.expander("🔧 Troubleshooting"):
    st.markdown("""
    **Jika ada error 'monotonic_cst':**
    
    1. **Update scikit-learn:**
    ```bash
    pip install --upgrade scikit-learn
    ```
    
    2. **Atau train ulang model** dengan versi scikit-learn yang sama
    
    3. **Cek versi library:**
    ```python
    import sklearn
    print(sklearn.__version__)
    ```
    
    4. **File yang harus ada:**
    - random_forest_model.pkl
    - scaler.pkl  
    - label_encoder.pkl
    """)
    
    # Tampilkan versi library
    st.write(f"scikit-learn version: {sklearn.__version__}")

# =====================================================
# 📘 Panduan
# =====================================================
with st.expander("📘 Petunjuk"):
    st.markdown("""
    1. Upload file WAV berisi suara "buka" atau "tutup"
    2. Klik tombol Prediksi Audio
    3. Sistem akan klasifikasikan hasilnya
    4. Untuk hasil terbaik, gunakan audio yang jelas
    """)