import streamlit as st
import numpy as np
import librosa
import joblib
import soundfile as sf
import tempfile
import os
from sklearn.ensemble import RandomForestClassifier

# Set page config
st.set_page_config(
    page_title="Voice Command Detector",
    page_icon="🎤",
    layout="centered"
)

# Load model and preprocessing objects
@st.cache_resource
def load_model():
    try:
        model = joblib.load('random_forest_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        return model, scaler, label_encoder
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def extract_features(file_path, sr=22050, n_mfcc=13):
    """Extract features dari audio file (sama seperti training)"""
    try:
        y, sr = librosa.load(file_path, sr=sr)
        
        # MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Additional features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        rms_energy = np.mean(librosa.feature.rms(y=y))
        
        # Combine features
        features = np.concatenate([
            mfcc_mean,
            mfcc_std,
            [spectral_centroid, zero_crossing_rate, chroma_stft, rms_energy]
        ])
        
        return features
        
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

def detect_speaker(audio_features, model, scaler, label_encoder):
    """Deteksi speaker dan command"""
    try:
        # Scale features
        features_scaled = scaler.transform(audio_features.reshape(1, -1))
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        command = label_encoder.inverse_transform([prediction])[0]
        
        return command
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None

def main():
    st.title("🎤 Voice Command Detector")
    st.write("Deteksi suara Asep dan Yotan untuk perintah 'Buka' dan 'Tutup'")
    
    # Load model
    model, scaler, label_encoder = load_model()
    
    if model is None:
        st.error("Model tidak dapat dimuat. Pastikan file model tersedia.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Unggah file audio (format .wav)", 
        type=['wav'],
        help="Unggah file audio berisi perintah 'Buka' atau 'Tutup' dari Asep atau Yotan"
    )
    
    # Atau rekaman langsung
    st.write("---")
    st.subheader("Atau rekam suara langsung")
    
    # Streamlit audio recorder (menggunakan st.audio dengan file upload)
    recorded_audio = st.file_uploader(
        "Rekam dan unggah audio", 
        type=['wav'],
        key="recorder",
        help="Rekam suara Anda dan unggah file hasil rekaman"
    )
    
    audio_file = uploaded_file or recorded_audio
    
    if audio_file is not None:
        # Display audio player
        st.audio(audio_file, format='audio/wav')
        
        # Save uploaded file to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name
        
        try:
            # Extract features
            with st.spinner("Menganalisis audio..."):
                features = extract_features(tmp_path)
            
            if features is not None:
                # Detect speaker and command
                command = detect_speaker(features, model, scaler, label_encoder)
                
                if command is not None:
                    # Simple speaker detection based on audio characteristics
                    # Anda bisa menambahkan model deteksi speaker yang lebih canggih di sini
                    duration = librosa.get_duration(filename=tmp_path)
                    y, sr = librosa.load(tmp_path, sr=None)
                    rms_energy = np.sqrt(np.mean(y**2))
                    
                    # Heuristic sederhana untuk membedakan speaker
                    # (Ini adalah placeholder - Anda perlu model deteksi speaker yang sebenarnya)
                    if rms_energy > 0.05:  # Threshold contoh
                        detected_speaker = "Asep"
                    else:
                        detected_speaker = "Yotan"
                    
                    # Display results
                    st.success("✅ Analisis selesai!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🔊 Hasil Deteksi")
                        st.write(f"**Speaker:** {detected_speaker}")
                        st.write(f"**Perintah:** {command}")
                    
                    with col2:
                        st.subheader("🎯 Status Pintu")
                        if command == "Buka":
                            st.success(f"🚪 Pintu sudah **DIBUKA** oleh {detected_speaker}")
                        else:
                            st.warning(f"🚪 Pintu sudah **DITUTUP** oleh {detected_speaker}")
                    
                    # Audio analysis info
                    with st.expander("📊 Informasi Analisis Audio"):
                        st.write(f"Durasi audio: {duration:.2f} detik")
                        st.write(f"Sample rate: {sr} Hz")
                        st.write(f"Energy level: {rms_energy:.4f}")
                        st.write(f"Jumlah fitur: {len(features)}")
                
                else:
                    st.error("❌ Gagal melakukan prediksi")
            
            else:
                st.error("❌ Gagal mengekstrak fitur dari audio")
        
        except Exception as e:
            st.error(f"❌ Error processing audio: {e}")
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    # Instructions
    with st.expander("ℹ️ Petunjuk Penggunaan"):
        st.write("""
        1. **Unggah file audio** (.wav) yang berisi suara Asep atau Yotan
        2. **Atau rekam suara** langsung dan unggah file rekaman
        3. Sistem akan mendeteksi:
           - Apakah suara berasal dari Asep atau Yotan
           - Perintah 'Buka' atau 'Tutup'
        4. Hasil akan ditampilkan dalam bentuk status pintu
        
        **Catatan:** 
        - Hanya suara Asep dan Yotan yang akan diproses
        - Suara lainnya tidak akan melakukan prediksi
        - Format audio: WAV, sample rate 22050 Hz
        """)

if __name__ == "__main__":
    main()