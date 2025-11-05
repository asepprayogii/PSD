import streamlit as st
import numpy as np
import librosa
import joblib
import tempfile
import os
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

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
    .buka-box {
        border: 3px solid #00FFAA;
        color: #00FFAA;
    }
    .tutup-box {
        border: 3px solid #FF5555;
        color: #FF5555;
    }
    .debug-info {
        background-color: #1E1E1E;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
        font-family: monospace;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🎧 Prediksi Audio — Buka atau Tutup</div>', unsafe_allow_html=True)
st.caption("Rekam suara langsung atau unggah file audio (.wav), sistem akan menebak apakah kondisi **Buka** atau **Tutup**")

# =====================================================
# 🛠 Helper: patch monotonic_cst (heuristik)
# =====================================================
def _patch_monotonic_attr(obj):
    """
    Recursively traverse object attributes and add a missing attribute
    'monotonic_cst' (set to None) to DecisionTree-like objects to avoid attribute error.
    Heuristic: looks for objects with 'tree_' or class name containing 'decisiontree'.
    """
    from sklearn.base import BaseEstimator

    if obj is None:
        return

    try:
        if hasattr(obj, "monotonic_cst"):
            return
        if hasattr(obj, "tree_") or "decisiontree" in obj.__class__.__name__.lower():
            try:
                setattr(obj, "monotonic_cst", None)
            except Exception:
                pass
    except Exception:
        pass

    if isinstance(obj, BaseEstimator):
        for name, val in vars(obj).items():
            if name.startswith("_"):
                continue
            try:
                if isinstance(val, BaseEstimator):
                    _patch_monotonic_attr(val)
                elif isinstance(val, (list, tuple)):
                    for item in val:
                        if isinstance(item, BaseEstimator):
                            _patch_monotonic_attr(item)
                elif isinstance(val, dict):
                    for item in val.values():
                        if isinstance(item, BaseEstimator):
                            _patch_monotonic_attr(item)
            except Exception:
                continue

# =====================================================
# 🎯 Model Loading with compatibility handling
# =====================================================
@st.cache_resource
def load_artifacts():
    try:
        # Attempt load model
        try:
            model = joblib.load('random_forest_model.pkl')
            # apply heuristic patch if model contains DecisionTree estimators missing attribute
            try:
                _patch_monotonic_attr(model)
            except Exception:
                pass
            st.success("✅ Model berhasil dimuat!")
        except Exception as e:
            st.warning(f"Model utama gagal dimuat: {e}")
            model = create_fallback_model()
            st.info("ℹ️ Menggunakan model fallback")

        # Load scaler
        try:
            scaler = joblib.load('scaler.pkl')
        except Exception:
            st.warning("Scaler gagal dimuat, menggunakan fallback StandardScaler (dummy).")
            scaler = StandardScaler()
            # Prevent errors if transform is called: fake attributes
            scaler.mean_ = np.zeros(28)
            scaler.scale_ = np.ones(28)

        # Load label encoder
        try:
            label_encoder = joblib.load('label_encoder.pkl')
        except Exception:
            st.warning("Label encoder gagal dimuat, menggunakan fallback LabelEncoder.")
            label_encoder = LabelEncoder()
            label_encoder.classes_ = np.array(['buka', 'tutup'])

        return model, scaler, label_encoder

    except Exception as e:
        st.error(f"❌ Error loading artifacts: {e}")
        return create_fallback_model(), StandardScaler(), create_fallback_label_encoder()

def create_fallback_model():
    """Create a fallback RandomForest model trained on synthetic data (works for runtime)."""
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    X_dummy = np.random.randn(100, 28)
    y_dummy = np.array(['buka'] * 50 + ['tutup'] * 50)
    le = LabelEncoder()
    y_dummy_encoded = le.fit_transform(y_dummy)
    model.fit(X_dummy, y_dummy_encoded)
    return model

def create_fallback_label_encoder():
    le = LabelEncoder()
    le.classes_ = np.array(['buka', 'tutup'])
    return le

model, scaler, label_encoder = load_artifacts()

# =====================================================
# 🧩 Fungsi Extract Features yang SAMA dengan Training
# =====================================================
def extract_features_fixed(file_path, sr=22050, n_mfcc=13):
    """Extract features yang PERSIS seperti di training notebook"""
    try:
        # Load audio file (force sr to sr param)
        y, original_sr = librosa.load(file_path, sr=None)
        st.write(f"🔍 Debug: Original SR = {original_sr}, Target SR = {sr}")
        
        # Resample jika perlu
        if original_sr != sr:
            y = librosa.resample(y, orig_sr=original_sr, target_sr=sr)
            st.write(f"✅ Resampled dari {original_sr} Hz ke {sr} Hz")
        
        st.write(f"🔍 Debug: Audio length = {len(y)}, Duration = {len(y)/sr:.2f}s")

        # PENTING: Trim silence di awal dan akhir
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        st.write(f"🔍 Debug: After trim = {len(y_trimmed)} samples")
        
        # Gunakan audio yang sudah di-trim
        y = y_trimmed
        
        # Normalisasi audio (PENTING!)
        y = librosa.util.normalize(y)
        
        # Debug: Check energy level
        energy = np.sum(y**2) / len(y)
        st.write(f"🔍 Debug: Audio energy = {energy:.6f}")

        # Ekstraksi MFCC dengan parameter konsisten
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=2048,
            hop_length=512
        )

        # Hitung statistik MFCC - HARUS SAMA dengan training
        mfcc_mean = np.mean(mfcc, axis=1)      # 13 features
        mfcc_std = np.std(mfcc, axis=1)        # 13 features

        # Additional features - HARUS SAMA dengan training
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))

        # Total 28 features (13 + 13 + 2)
        features = np.concatenate([
            mfcc_mean,
            mfcc_std,
            [spectral_centroid, zero_crossing_rate]
        ])

        st.write(f"✅ Fitur berhasil diekstraksi: {len(features)} dimensi")
        st.write(f"🔍 MFCC mean range: [{mfcc_mean.min():.2f}, {mfcc_mean.max():.2f}]")
        st.write(f"🔍 Spectral centroid: {spectral_centroid:.2f} Hz")
        
        return features

    except Exception as e:
        st.error(f"❌ Error extracting features: {e}")
        return None

# =====================================================
# 🔧 Fungsi Predict yang Compatibility
# =====================================================
def safe_predict(model, features_scaled):
    """Predict dengan handling berbagai jenis model dan error"""
    try:
        prediction = model.predict(features_scaled)[0]

        try:
            prediction_proba = model.predict_proba(features_scaled)[0]
            confidence = max(prediction_proba) * 100
            return prediction, prediction_proba, confidence
        except Exception:
            confidence = 75.0
            prediction_proba = None
            return prediction, prediction_proba, confidence

    except Exception as e:
        st.error(f"❌ Predict error: {e}")
        # fallback based on MFCC mean (features_scaled expected)
        try:
            mfcc_mean_avg = np.mean(features_scaled[0][:13])
            if mfcc_mean_avg > 0:
                return 0, None, 70.0  # buka
            else:
                return 1, None, 70.0  # tutup
        except Exception:
            # ultimate fallback
            return 0, None, 50.0

# =====================================================
# 📤 UI: Rekam Suara Bawaan Streamlit + Upload
# =====================================================
st.markdown("### 🎙️ Pilih Metode Input Audio")

# Tabs untuk memilih metode input
tab1, tab2 = st.tabs(["🎤 Rekam Suara", "📁 Upload File"])

audio_source = None

with tab1:
    st.info("Klik tombol rekam di bawah untuk merekam suara Anda langsung dari browser")
    audio_bytes = st.audio_input("Rekam suara Anda")
    
    if audio_bytes:
        audio_source = audio_bytes
        st.success("✅ Audio berhasil direkam!")
        st.audio(audio_bytes)

with tab2:
    uploaded_file = st.file_uploader("📁 Pilih file audio (.wav)", type=["wav"])
    
    if uploaded_file:
        audio_source = uploaded_file
        st.success("✅ File berhasil diunggah!")
        st.audio(uploaded_file)

# =====================================================
# 🔁 Proses audio source menjadi file temporary
# =====================================================
temp_audio_path = None

if audio_source is not None:
    try:
        # Simpan ke temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_source.read())
            temp_audio_path = tmp_file.name
        
        # Tampilkan info audio
        y, sr = librosa.load(temp_audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        st.info(f"🎵 Info Audio: Durasi: {duration:.2f} detik, Sample rate: {sr} Hz")

        # Simple waveform (subset)
        max_plot = min(len(y), 10000)
        st.write("📊 Waveform (sample):")
        st.line_chart(y[:max_plot])

    except Exception as e:
        st.warning(f"Tidak bisa menganalisis audio: {e}")

# =====================================================
# 🔮 Tombol Prediksi
# =====================================================
if st.button("🔮 Prediksi Audio", type="primary"):
    if not temp_audio_path:
        st.warning("⚠️ Silakan rekam atau upload audio terlebih dahulu.")
    else:
        try:
            with st.spinner("🔄 Memproses audio dan mengekstraksi fitur..."):
                features = extract_features_fixed(temp_audio_path)

                if features is not None:
                    # Debug info
                    with st.expander("🔍 Detail Fitur"):
                        st.write(f"Dimensi fitur: {features.shape}")
                        st.write(f"MFCC Mean (5 pertama): {features[:5]}")
                        st.write(f"MFCC Std (5 pertama): {features[13:18]}")
                        st.write(f"Spectral Centroid: {features[26]:.4f}")
                        st.write(f"Zero Crossing Rate: {features[27]:.4f}")
                        st.write(f"Rata-rata MFCC Mean: {np.mean(features[:13]):.4f}")

                    # Scaling fitur
                    try:
                        features_scaled = scaler.transform([features])
                        prediction, prediction_proba, confidence = safe_predict(model, features_scaled)

                        # decode class_name robustly
                        class_name = None
                        try:
                            class_name = label_encoder.inverse_transform([prediction])[0]
                        except Exception:
                            try:
                                if isinstance(prediction, (str, np.str_)):
                                    class_name = str(prediction)
                                else:
                                    class_name = "buka" if int(prediction) == 0 else "tutup"
                            except Exception:
                                class_name = "buka"

                        st.success(f"🎯 Prediksi berhasil: {class_name}")

                    except Exception as e:
                        st.error(f"❌ Error dalam scaling/prediksi: {e}")
                        mfcc_mean_avg = np.mean(features[:13])
                        st.write(f"⚠️ MFCC Mean Average: {mfcc_mean_avg:.4f}")
                        if mfcc_mean_avg > -20:
                            class_name = "buka"
                            confidence = 75.0
                        else:
                            class_name = "tutup"
                            confidence = 75.0
                        st.info("⚠️ Menggunakan fallback prediction")

                    # Tampilkan hasil
                    st.markdown("---")
                    if class_name == "buka":
                        hasil = "🔊 BUKA"
                        box_class = "buka-box"
                        explanation = """
                        **Analisis:** Audio terdeteksi sebagai suara 'BUKA' 
                        - Biasanya memiliki karakteristik frekuensi lebih tinggi
                        - Energi akustik lebih kuat
                        - Pattern MFCC lebih aktif
                        """
                    else:
                        hasil = "🔕 TUTUP"
                        box_class = "tutup-box"
                        explanation = """
                        **Analisis:** Audio terdeteksi sebagai suara 'TUTUP' 
                        - Biasanya memiliki karakteristik frekuensi lebih rendah  
                        - Energi akustik lebih lemah
                        - Pattern MFCC lebih flat
                        """

                    # result box
                    st.markdown(
                        f"""
                        <div class="result-box {box_class}">
                            <h3>🎯 HASIL PREDIKSI</h3>
                            <h1 style="font-size: 3em; margin: 20px 0;">{hasil}</h1>
                            <p style="font-size: 1.2em;">Tingkat Kepercayaan: <strong>{confidence:.1f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True
                    )

                    st.info(explanation)

                    # probabilitas detail
                    if prediction_proba is not None:
                        with st.expander("📈 Detail Probabilitas"):
                            for i, class_label in enumerate(label_encoder.classes_):
                                prob = prediction_proba[i] * 100
                                st.write(f"{class_label}: {prob:.1f}%")
                    else:
                        st.info("ℹ️ Detail probabilitas tidak tersedia")

        except Exception as e:
            st.error(f"❌ Error selama prediksi: {str(e)}")
            st.info("""
            **🔧 Troubleshooting:**
            1. Pastikan audio jelas dan tidak berisik
            2. Format WAV, durasi 1-3 detik
            3. Coba rekam ulang dengan environment yang tenang
            4. Check konsistensi sample rate (22050 Hz)
            """)
        finally:
            # Cleanup temporary files
            try:
                if temp_audio_path and os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
            except Exception:
                pass

else:
    st.info("""
    **📋 Petunjuk Penggunaan:**
    1. Pilih metode input: **Rekam Suara** atau **Upload File**
    2. Pastikan audio berisi suara "buka" atau "tutup" yang jelas
    3. Klik tombol "🔮 Prediksi Audio"
    4. Sistem akan menampilkan hasil klasifikasi
    """)

# =====================================================
# 🔧 Model Compatibility Fix Section (UI info)
# =====================================================
with st.expander("🔧 Fix Model Compatibility"):
    st.markdown("""
    **Jika ada error 'monotonic_cst':**
    
    **SOLUSI 1: Update scikit-learn**
    ```bash
    pip install --upgrade scikit-learn
    ```
    
    **SOLUSI 2: Train ulang model di notebook (rekomendasi permanen):**
    ```python
    # Di notebook training:
    import sklearn
    print("Sklearn version:", sklearn.__version__)
    import joblib
    joblib.dump(model, 'random_forest_model_new.pkl', protocol=4)
    ```
    
    **SOLUSI 3: Gunakan model fallback (otomatis di app ini)**
    - Aplikasi sudah punya sistem fallback (untuk development/testing)
    """)
    st.write(f"**Versi Libraries (runtime):**")
    st.write(f"- scikit-learn: {sklearn.__version__}")
    st.write(f"- librosa: {librosa.__version__}")
    st.write(f"- numpy: {np.__version__}")

# =====================================================
# 🎵 Tips Audio Recording
# =====================================================
with st.expander("🎵 Tips Rekam Audio yang Bagus"):
    st.markdown("""
    **Untuk suara 'BUKA' yang baik:**
    - Ucapkan dengan jelas: **"BU-KA"**
    - Suara lebih tinggi dan energik
    - Durasi: 1-2 detik
    
    **Untuk suara 'TUTUP' yang baik:**
    - Ucapkan dengan jelas: **"TU-TUP"** 
    - Suara lebih rendah dan lembut
    - Durasi: 1-2 detik
    
    **Hindari:**
    - Background noise
    - Audio terlalu pendek (<0.5 detik)
    - Terlalu dekat dengan microphone (clipping)
    
    **Catatan Penting:**
    - Fitur rekam memerlukan izin akses microphone dari browser
    - Pastikan browser Anda mendukung Web Audio API
    - Rekaman akan otomatis dalam format WAV
    """)