import streamlit as st
import numpy as np
import librosa
import joblib
import tempfile
import os

# Set page config
st.set_page_config(
    page_title="Voice Command Detector",
    page_icon="🎤",
    layout="centered"
)

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
    try:
        y, sr = librosa.load(file_path, sr=sr)
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        rms_energy = np.mean(librosa.feature.rms(y=y))
        
        features = np.concatenate([
            mfcc_mean,
            mfcc_std,
            [spectral_centroid, zero_crossing_rate, chroma_stft, rms_energy]
        ])
        
        return features
        
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

def detect_speaker_advanced(audio_path):
    """Deteksi speaker berdasarkan karakteristik audio yang lebih advance"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        
        # Ekstrak fitur untuk deteksi speaker
        duration = librosa.get_duration(y=y, sr=sr)
        rms_energy = np.sqrt(np.mean(y**2))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Analisis pitch/frekuensi dasar
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        
        # Heuristic rules berdasarkan analisis suara
        # (Ini bisa disesuaikan dengan karakteristik suara Asep & Yotan Anda)
        
        # Rule 1: Berdasarkan energy (suara keras/lembut)
        # Rule 2: Berdasarkan pitch (suara tinggi/rendah)
        # Rule 3: Berdasarkan spectral centroid (bright/dark voice)
        
        st.write("🔍 **Analisis Karakteristik Suara:**")
        st.write(f"   - Energy: {rms_energy:.4f}")
        st.write(f"   - Pitch rata-rata: {pitch_mean:.1f} Hz")
        st.write(f"   - Spectral Centroid: {spectral_centroid:.1f}")
        st.write(f"   - Zero Crossing Rate: {zero_crossing_rate:.4f}")
        
        # LOGIC DETEKSI SPEAKER (SESUAIKAN DENGAN DATA ANDA)
        # Contoh heuristic - sesuaikan threshold berdasarkan data nyata
        
        if rms_energy > 0.03 and pitch_mean < 180:
            return "Asep"
        elif rms_energy <= 0.03 and pitch_mean >= 180:
            return "Yotan"
        else:
            # Fallback berdasarkan kombinasi fitur
            if spectral_centroid > 1500 and zero_crossing_rate > 0.08:
                return "Asep"
            else:
                return "Yotan"
                
    except Exception as e:
        st.error(f"Error in speaker detection: {e}")
        return "Tidak Diketahui"

def main():
    st.title("🎤 Voice Command Detector")
    st.write("Deteksi perintah 'Buka' dan 'Tutup' + identifikasi speaker")
    
    # Load model
    model, scaler, label_encoder = load_model()
    
    if model is None:
        st.error("Model tidak dapat dimuat.")
        return
    
    # Tampilkan info model
    st.write("---")
    st.subheader("🔍 Model Info")
    st.write(f"**Model bisa deteksi:** {list(label_encoder.classes_)}")
    st.info("ℹ️ Model ini hanya bisa membedakan **KATA** (Buka/Tutup). Speaker dideteksi secara terpisah.")
    
    # Audio recorder
    st.write("---")
    st.subheader("🎙️ Rekam Suara")
    
    audio_bytes = st.audio_input("Rekam suara Anda (ucapkan 'Buka' atau 'Tutup'):")
    
    if audio_bytes is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes.getvalue())
            tmp_path = tmp_file.name
        
        st.audio(audio_bytes, format='audio/wav')
        
        col1, col2 = st.columns(2)
        with col1:
            predict_command = st.button("🎯 **Deteksi Perintah**", type="primary", use_container_width=True)
        with col2:
            detect_speaker = st.button("👤 **Deteksi Speaker**", type="secondary", use_container_width=True)
        
        if predict_command:
            try:
                with st.spinner("Menganalisis perintah..."):
                    features = extract_features(tmp_path)
                
                if features is not None:
                    features_scaled = scaler.transform(features.reshape(1, -1))
                    prediction = model.predict(features_scaled)[0]
                    predicted_command = label_encoder.inverse_transform([prediction])[0]
                    
                    # Show probabilities
                    probabilities = model.predict_proba(features_scaled)[0]
                    
                    st.write("---")
                    st.subheader("📊 Hasil Deteksi Perintah")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Probabilitas:**")
                        for i, (class_name, prob) in enumerate(zip(label_encoder.classes_, probabilities)):
                            color = "🟢" if prob == max(probabilities) else "⚪"
                            st.write(f"{color} {class_name}: {prob:.3f} ({prob*100:.1f}%)")
                    
                    with col2:
                        st.write("**Hasil:**")
                        if predicted_command == "Buka":
                            st.success(f"**Perintah: {predicted_command}** 🚪")
                            st.balloons()
                        else:
                            st.warning(f"**Perintah: {predicted_command}** 🚪")
                
                else:
                    st.error("Gagal mengekstrak fitur audio")
            
            except Exception as e:
                st.error(f"Error: {e}")
        
        if detect_speaker:
            try:
                with st.spinner("Menganalisis speaker..."):
                    detected_speaker = detect_speaker_advanced(tmp_path)
                
                st.write("---")
                st.subheader("👤 Hasil Deteksi Speaker")
                
                if detected_speaker == "Asep":
                    st.success(f"**Speaker: {detected_speaker}** 🔊")
                else:
                    st.info(f"**Speaker: {detected_speaker}** 🔊")
                    
            except Exception as e:
                st.error(f"Error deteksi speaker: {e}")
        
        # Tombol kombinasi
        st.write("---")
        if st.button("🚀 **DETEKSI LENGKAP**", type="primary"):
            try:
                with st.spinner("Analisis lengkap..."):
                    # Deteksi perintah
                    features = extract_features(tmp_path)
                    features_scaled = scaler.transform(features.reshape(1, -1))
                    prediction = model.predict(features_scaled)[0]
                    predicted_command = label_encoder.inverse_transform([prediction])[0]
                    
                    # Deteksi speaker
                    detected_speaker = detect_speaker_advanced(tmp_path)
                
                st.write("---")
                st.subheader("🎯 HASIL LENGKAP")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Deteksi Perintah:**")
                    if predicted_command == "Buka":
                        st.success(f"**{predicted_command}** 🚪")
                    else:
                        st.warning(f"**{predicted_command}** 🚪")
                
                with col2:
                    st.write("**Deteksi Speaker:**")
                    if detected_speaker == "Asep":
                        st.success(f"**{detected_speaker}** 🔊")
                    else:
                        st.info(f"**{detected_speaker}** 🔊")
                
                # Final result
                st.write("---")
                st.subheader("🔔 STATUS PINTU")
                if predicted_command == "Buka":
                    st.success(f"🚪 **Pintu sudah DIBUKA oleh {detected_speaker}**")
                    st.balloons()
                else:
                    st.warning(f"🚪 **Pintu sudah DITUTUP oleh {detected_speaker}**")
                    
            except Exception as e:
                st.error(f"Error analisis lengkap: {e}")
        
        # Cleanup
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    
    # Instructions
    with st.expander("ℹ️ Petunjuk Penggunaan"):
        st.write("""
        **Cara penggunaan:**
        1. **Rekam suara** - ucapkan "Buka" atau "Tutup" dengan jelas
        2. **Pilih analisis:**
           - 🎯 **Deteksi Perintah**: Hanya deteksi kata "Buka"/"Tutup"
           - 👤 **Deteksi Speaker**: Hanya deteksi speaker Asep/Yotan  
           - 🚀 **Deteksi Lengkap**: Deteksi perintah + speaker
        3. **Lihat hasil** dan status pintu
        
        **Catatan:** 
        - Model saat ini hanya terlatih untuk deteksi **KATA**
        - Deteksi speaker menggunakan heuristic sederhana
        - Untuk akurasi lebih tinggi, perlu model deteksi speaker terpisah
        """)

if __name__ == "__main__":
    main()