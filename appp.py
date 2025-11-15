import streamlit as st
import os
import tempfile
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import joblib
import traceback

# Fix untuk error ComplexWarning
try:
    from numpy.core.numeric import ComplexWarning
except ImportError:
    class ComplexWarning(UserWarning):
        pass
    import numpy
    numpy.ComplexWarning = ComplexWarning

import librosa

warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Audio Classification: Buka/Tutup & Speaker ID",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling dengan animasi pintu
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* AKSES DITOLAK */
    .access-denied {
        background: linear-gradient(135deg, #ff6b6b, #ee5a6f);
        padding: 3rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 5px 20px rgba(255,0,0,0.3);
    }
    
    .access-denied h1 {
        font-size: 3rem;
        margin: 0;
    }
    
    /* ANIMASI PINTU - Door that moves to close/open */
    .door-scene {
        background: linear-gradient(to bottom, #87ceeb 0%, #e0f6ff 50%, #c9a87c 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        position: relative;
        height: 450px;
        overflow: hidden;
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
    }
    
    .wall {
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 350px;
        background: linear-gradient(to bottom, #f5f5dc, #d3d3d3);
    }
    
    .door-opening {
        position: absolute;
        bottom: 50px;
        left: 50%;
        transform: translateX(-50%);
        width: 200px;
        height: 280px;
        background: #333;
        border-radius: 10px 10px 0 0;
        box-shadow: inset 0 0 30px rgba(0,0,0,0.5);
    }
    
    .door-left, .door-right {
        position: absolute;
        bottom: 50px;
        width: 100px;
        height: 280px;
        background: linear-gradient(135deg, #8B4513 0%, #A0522D 50%, #8B4513 100%);
        border: 3px solid #654321;
        transition: all 2s ease-in-out;
        box-shadow: 0 5px 20px rgba(0,0,0,0.4);
    }
    
    .door-left {
        left: calc(50% - 100px);
        border-radius: 8px 0 0 0;
    }
    
    .door-right {
        left: 50%;
        border-radius: 0 8px 0 0;
    }
    
    /* Animasi membuka - dengan keyframes */
    .door-left.door-open {
        animation: openDoorLeft 2s ease-in-out forwards;
    }
    
    .door-right.door-open {
        animation: openDoorRight 2s ease-in-out forwards;
    }
    
    @keyframes openDoorLeft {
        0% { transform: translateX(0); }
        100% { transform: translateX(-110%); }
    }
    
    @keyframes openDoorRight {
        0% { transform: translateX(0); }
        100% { transform: translateX(110%); }
    }
    
    /* Animasi menutup - tambah efek slam */
    .door-left.door-close {
        animation: closeDoorLeft 2s ease-in-out forwards;
    }
    
    .door-right.door-close {
        animation: closeDoorRight 2s ease-in-out forwards;
    }
    
    @keyframes closeDoorLeft {
        0% { transform: translateX(-110%); }
        80% { transform: translateX(5%); }
        100% { transform: translateX(0); }
    }
    
    @keyframes closeDoorRight {
        0% { transform: translateX(110%); }
        80% { transform: translateX(-5%); }
        100% { transform: translateX(0); }
    }
    
    .door-handle {
        position: absolute;
        top: 50%;
        transform: translateY(-50%);
        width: 12px;
        height: 35px;
        background: linear-gradient(to bottom, #FFD700, #DAA520);
        border-radius: 6px;
        box-shadow: 2px 2px 4px rgba(0,0,0,0.4);
    }
    
    .door-left .door-handle {
        right: 15px;
    }
    
    .door-right .door-handle {
        left: 15px;
    }
    
    .door-lock {
        position: absolute;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        font-size: 5rem;
        z-index: 10;
        text-shadow: 0 0 10px rgba(0,0,0,0.5);
    }
    
    .character {
        position: absolute;
        bottom: 50px;
        left: 50%;
        transform: translateX(-50%);
        font-size: 120px;
        opacity: 0;
        transition: opacity 1.5s ease-in-out 2s;
        z-index: 5;
        text-shadow: 0 5px 10px rgba(0,0,0,0.3);
        display: none;
    }
    
    .character-show {
        opacity: 1;
        display: none;
    }
    
    .action-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin: 1rem 0;
        padding: 1rem;
        border-radius: 10px;
    }
    
    .action-buka {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        color: #155724;
    }
    
    .action-tutup {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        color: #721c24;
    }
    
    .speaker-badge {
        display: inline-block;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-size: 1.5rem;
        font-weight: bold;
        color: white;
        margin: 1rem;
    }
    
    .speaker-asep {
        background: linear-gradient(135deg, #4facfe, #00f2fe);
    }
    
    .speaker-yotan {
        background: linear-gradient(135deg, #fa709a, #fee140);
    }
    
    .stats-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

def load_models():
    """Load models dengan error handling"""
    try:
        model_jenis = joblib.load('model_results/rf_model_buka_tutup.pkl')
        model_speaker = joblib.load('model_results/rf_model_speaker.pkl')
        return model_jenis, model_speaker, True, "Models loaded successfully"
    except Exception as e:
        return None, None, False, f"Error loading models: {str(e)}"

def extract_comprehensive_features(y, sr=22050):
    """Ekstraksi fitur audio - SAMA PERSIS dengan notebook"""
    feats = {}
    feats['stat_mean'] = np.mean(y)
    feats['stat_std'] = np.std(y)
    feats['stat_skew'] = stats.skew(y)
    feats['stat_kurt'] = stats.kurtosis(y)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        feats[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
        feats[f'mfcc_{i}_std'] = np.std(mfccs[i])

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(12):
        feats[f'chroma_{i}_mean'] = np.mean(chroma[i])
        feats[f'chroma_{i}_std'] = np.std(chroma[i])

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=10)
    for i in range(10):
        feats[f'mel_{i}_mean'] = np.mean(mel_spec[i])
        feats[f'mel_{i}_std'] = np.std(mel_spec[i])

    feats['spec_centroid_mean'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    feats['spec_rolloff_mean'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    feats['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))

    return feats

def predict_audio_exact(audio_path, model_jenis, model_speaker, confidence_threshold=0.55, command_threshold=0.55):
    """
    Prediksi audio dengan validasi lengkap:
    1. Cek speaker dulu (threshold 55%)
    2. Cek apakah benar-benar ngomong buka/tutup (threshold 55%)
    3. Jika salah satu gagal → AKSES DITOLAK
    """
    # Load dan preprocessing audio
    y, sr = librosa.load(audio_path, sr=22050)
    y = y / (np.max(np.abs(y)) + 1e-6)
    y[np.abs(y) < 0.005] = 0.0

    max_len = int(sr * 2.0)
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)), mode='constant')
    else:
        start = (len(y) - max_len) // 2
        y = y[start:start+max_len]

    # Ekstrak fitur
    features = extract_comprehensive_features(y, sr)
    feature_order = model_jenis.feature_names_in_
    X_new = pd.DataFrame([features])[feature_order]

    # STEP 1: CEK SPEAKER
    pred_proba_speaker = model_speaker.predict_proba(X_new)[0]
    max_speaker_conf = max(pred_proba_speaker)
    
    if max_speaker_conf < confidence_threshold:
        return {
            'success': True,
            'voice_detected': False,
            'reason': 'speaker_unknown',
            'speaker_confidence': max_speaker_conf * 100
        }
    
    pred_label_speaker = model_speaker.predict(X_new)[0]
    
    # STEP 2: CEK COMMAND (BUKA/TUTUP)
    pred_label_buka = model_jenis.predict(X_new)[0]
    pred_proba_buka = model_jenis.predict_proba(X_new)[0]
    max_command_conf = max(pred_proba_buka)
    
    # VALIDASI: Apakah benar-benar ngomong buka/tutup?
    if max_command_conf < command_threshold:
        return {
            'success': True,
            'voice_detected': False,
            'reason': 'no_valid_command',
            'command_confidence': max_command_conf * 100,
            'speaker': pred_label_speaker
        }

    # SEMUA VALIDASI LOLOS
    return {
        'jenis': {
            'prediction': pred_label_buka,
            'probability': pred_proba_buka,
            'confidence': max_command_conf * 100,
            'probabilities': dict(zip(model_jenis.classes_, pred_proba_buka))
        },
        'speaker': {
            'prediction': pred_label_speaker,
            'probability': pred_proba_speaker,
            'confidence': max_speaker_conf * 100,
            'probabilities': dict(zip(model_speaker.classes_, pred_proba_speaker)),
            'is_unknown': False
        },
        'features': features,
        'success': True,
        'voice_detected': True
    }

def display_door_animation(action, speaker):
    """Animasi pintu dobel yang membuka/menutup - TANPA karakter icon"""
    is_open = "buka" in action.lower()
    door_class = "door-open" if is_open else "door-close"
    
    lock_icon = "" if is_open else "🔒"
    
    st.markdown(f"""
    <div class="door-scene">
        <div class="wall">
            <div class="door-opening"></div>
            <div class="door-left {door_class}">
                <div class="door-handle"></div>
            </div>
            <div class="door-right {door_class}">
                <div class="door-handle"></div>
            </div>
            <div class="door-lock">{lock_icon}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🎵 Sistem Klasifikasi Audio Pintu</h1>', unsafe_allow_html=True)
    
    with st.spinner("Memuat model..."):
        model_jenis, model_speaker, models_loaded, load_message = load_models()
    
    if not models_loaded:
        st.error(f"❌ {load_message}")
        st.info("Pastikan file model ada di folder `model_results/`")
        return
    
    st.success("✅ Model siap digunakan")
    
    # SIDEBAR
    with st.sidebar:
        st.header("ℹ️ Informasi Sistem")
        st.info("**Menerima 2 speaker:**\n\n Asep\n\n Yotan")
        st.markdown("---")

    # Upload/Record tabs
    tab1, tab2 = st.tabs(["📁 Upload File", "🎤 Rekam Suara"])
    
    with tab1:
        st.header("📁 Upload File Audio")
        uploaded_file = st.file_uploader(
            "Pilih file audio (WAV, MP3)",
            type=['wav', 'mp3']
        )
    
    with tab2:
        st.header("🎤 Rekam Suara")
        audio_rec = st.audio_input("Rekam suara:")
    
    audio_file = uploaded_file or audio_rec
    
    if audio_file is not None:
        st.success(f"✅ Audio: {audio_file.name if hasattr(audio_file, 'name') else 'recording.wav'}")
        st.audio(audio_file, format='audio/wav')
        
        if st.button("🎯 ANALISIS", type="primary", use_container_width=True):
            with st.spinner("🔍 Menganalisis..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    if hasattr(audio_file, 'getvalue'):
                        tmp_file.write(audio_file.getvalue())
                    elif hasattr(audio_file, 'read'):
                        tmp_file.write(audio_file.read())
                    temp_path = tmp_file.name
                
                try:
                    # PREDIKSI dengan double threshold
                    result = predict_audio_exact(
                        temp_path, 
                        model_jenis, 
                        model_speaker, 
                        confidence_threshold=0.55,  # Speaker threshold
                        command_threshold=0.55      # Command threshold
                    )
                    
                    if result and result.get('success', False):
                        
                        # CEK: Apakah lolos semua validasi?
                        if not result.get('voice_detected', False):
                            # ❌ AKSES DITOLAK - TAMPILKAN CONFIDENCE
                            reason = result.get('reason', 'unknown')
                            
                            if reason == 'speaker_unknown':
                                message = "Suara tidak dikenali dalam sistem"
                                detail = "Hanya <b>Asep</b> dan <b>Yotan</b> yang terdaftar"
                                conf_value = result.get('speaker_confidence', 0)
                                conf_label = "Speaker Confidence"
                            else:
                                message = "Tidak ada perintah yang valid"
                                detail = "Silakan ucapkan <b>'BUKA'</b> atau <b>'TUTUP'</b> dengan jelas"
                                conf_value = result.get('command_confidence', 0)
                                conf_label = "Command Confidence"
                            
                            st.markdown(f"""
                            <div class="access-denied">
                                <h1>🚫 AKSES DITOLAK</h1>
                                <p style="font-size: 1.5rem; margin-top: 1rem;">
                                    {message}
                                </p>
                                <p style="font-size: 1.2rem;">
                                    {detail}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Tampilkan analisis confidence
                            st.error("⚠️ Validasi gagal")
                            
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                st.markdown(f"""
                                <div class="stats-box">
                                    <h4>{conf_label}</h4>
                                    <p style="font-size: 2rem; font-weight: bold; color: #dc3545;">
                                        {conf_value:.1f}%
                                    </p>
                                    <p style="color: #6c757d;">Minimum diperlukan: 
                                        {'55%' if reason == 'speaker_unknown' else '55%'}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                if conf_value < 40:
                                    st.warning("⚠️ Confidence sangat rendah - Audio mungkin tidak jelas atau bukan suara manusia")
                                elif conf_value < 55:
                                    st.info("ℹ️ Confidence mendekati threshold - Coba ucapkan lebih jelas")
                                else:
                                    st.info("ℹ️ Confidence hampir cukup - Sedikit lagi!")
                        
                        else:
                            # ✅ AKSES DITERIMA
                            action = result['jenis']['prediction']
                            speaker = result['speaker']['prediction']
                            
                            # Judul aksi
                            action_class = "action-buka" if "buka" in action.lower() else "action-tutup"
                            action_icon = "🔓" if "buka" in action.lower() else "🔒"
                            
                            st.markdown(f"""
                            <div class="action-title {action_class}">
                                {action_icon} PINTU {action.upper()} {action_icon}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Speaker badge
                            speaker_class = "speaker-asep" if "asep" in speaker.lower() else "speaker-yotan"
                            speaker_icon = "👨" if "asep" in speaker.lower() else "👨"
                            
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                st.markdown(f"""
                                <div style="text-align: center;">
                                    <div class="speaker-badge {speaker_class}">
                                        {speaker_icon} Oleh: {speaker.upper()}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # ANIMASI PINTU
                            display_door_animation(action, speaker)
                            
                            st.markdown("---")
                            
                            # Detail Results
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("🚪 Klasifikasi Perintah")
                                st.markdown(f"""
                                <div class="stats-box">
                                    <p style="font-size: 1.2rem; font-weight: bold;">
                                        Confidence: {result['jenis']['confidence']:.1f}%
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                for cls, prob in result['jenis']['probabilities'].items():
                                    st.progress(float(prob), text=f"{cls}: {prob:.1%}")
                            
                            with col2:
                                st.subheader("👤 Identifikasi Speaker")
                                st.markdown(f"""
                                <div class="stats-box">
                                    <p style="font-size: 1.2rem; font-weight: bold;">
                                        Confidence: {result['speaker']['confidence']:.1f}%
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                for cls, prob in result['speaker']['probabilities'].items():
                                    st.progress(float(prob), text=f"{cls}: {prob:.1%}")
                            
                            # Audio Features - Optional
                            with st.expander("🔬 Lihat Fitur Audio"):
                                features = result['features']
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.markdown("**Statistik:**")
                                    for k in ['stat_mean', 'stat_std', 'stat_skew', 'stat_kurt']:
                                        if k in features:
                                            st.text(f"{k}: {features[k]:.4f}")
                                
                                with col2:
                                    st.markdown("**MFCC (sample):**")
                                    count = 0
                                    for k, v in features.items():
                                        if 'mfcc' in k and count < 5:
                                            st.text(f"{k}: {v:.4f}")
                                            count += 1
                                
                                with col3:
                                    st.markdown("**Spectral:**")
                                    for k in ['spec_centroid_mean', 'spec_rolloff_mean', 'zero_crossing_rate']:
                                        if k in features:
                                            st.text(f"{k}: {features[k]:.4f}")
                                
                                st.info(f"📊 Total fitur: {len(features)}")
                    
                    else:
                        st.error("❌ Gagal prediksi")
                        
                except Exception as e:
                    st.error(f"⚠️ Error: {str(e)}")
                    with st.expander("Detail Error"):
                        st.code(traceback.format_exc())
                
                finally:
                    try:
                        os.unlink(temp_path)
                    except:
                        pass

if __name__ == "__main__":
    main()