import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# === Konfigurasi ===
MODEL_PATH = 'ECG200/best_model.pkl'
SCALER_PATH = 'ECG200/scaler.pkl'
EXPECTED_LENGTH = 96

# === Muat model dan scaler ===
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

try:
    model, scaler = load_model()
except Exception as e:
    st.error(f"❌ Gagal memuat model/scaler: {e}")
    st.stop()

# === Fungsi: Buat contoh file (1 baris x 96 kolom) ===
def create_sample_file():
    t = np.linspace(0, 4 * np.pi, EXPECTED_LENGTH)
    sample = np.sin(t) + 0.1 * np.random.randn(EXPECTED_LENGTH)
    df = pd.DataFrame([sample])
    return df.to_csv(index=False, header=False).encode('utf-8')

# === Sidebar ===
st.sidebar.title("🧭 Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Unggah Data", "Tentang"])

# === HALAMAN: Tentang ===
if page == "Tentang":
    st.title("ℹ️ Tentang Aplikasi & Medis")
    
    st.subheader("Dataset ECG200")
    st.write("""
    Dataset ini berisi 200 rekaman ECG (100 latih, 100 uji), masing-masing terdiri dari 96 titik waktu.
    Setiap sampel mewakili satu denyut jantung, dan diberi label:
    - **Kelas 1**: Normal
    - **Kelas -1**: Abnormal – kemungkinan **Infark Miokard (Myocardial Infarction)**
    """)
    
    st.subheader("Apa Itu Infark Miokard?")
    st.write("""
    **Infark Miokard** (serangan jantung) terjadi ketika aliran darah ke bagian otot jantung **terhambat**, 
    biasanya karena penyumbatan pada arteri koroner oleh plak lemak (aterosklerosis).

    **Dampak pada ECG**:
    - Gelombang **T** menjadi lebih tinggi atau terbalik,
    - Segmen **ST** bisa naik (elevasi ST) atau turun,
    - Gelombang **Q** patologis mungkin muncul.

    **Gejala umum**:
    - Nyeri dada seperti ditekan,
    - Sesak napas,
    - Mual, keringat dingin,
    - Nyeri menyebar ke lengan kiri, leher, atau rahang.

    ⚠️ **Catatan**: Aplikasi ini **bukan alat diagnosis medis**. Hasil hanya untuk edukasi dan eksperimen akademik.
    """)

# === HALAMAN: Unggah Data ===
else:
    st.title("🫀 Deteksi Kelainan Jantung dari Sinyal ECG")
    st.markdown("""
    Unggah file `.csv` berisi **96 nilai ECG** dalam **satu baris** (96 kolom) atau **satu kolom** (96 baris).
    """)

    # Unduh contoh
    sample_csv = create_sample_file()
    st.download_button(
        label="📄 Unduh Contoh File (1 baris × 96 kolom)",
        data=sample_csv,
        file_name="ecg_sample.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None)

            # Deteksi format
            if df.shape == (1, 96):
                signal = df.values.flatten()
            elif df.shape == (96, 1):
                signal = df.iloc[:, 0].values
            else:
                st.error(f"❌ Format harus 1×96 atau 96×1. File Anda: {df.shape[0]}×{df.shape[1]}")
                st.stop()

            if len(signal) != EXPECTED_LENGTH:
                st.error("❌ Jumlah nilai bukan 96.")
                st.stop()

            # Visualisasi input
            st.success("✅ Data valid! Visualisasi sinyal ECG:")
            fig, ax = plt.subplots(figsize=(9, 3))
            ax.plot(signal, color='black', linewidth=1.5)
            ax.set_title("Sinyal ECG Input", fontsize=12)
            ax.set_xlabel("Waktu (96 titik)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # Prediksi
            if st.button("🔍 Prediksi"):
                signal_scaled = scaler.transform(signal.reshape(1, -1))
                pred = model.predict(signal_scaled)[0]

                # Visualisasi hasil + penjelasan
                st.markdown("### 📊 Hasil Prediksi")
                fig2, ax2 = plt.subplots(figsize=(9, 3))
                color = 'green' if pred == 1 else 'red'
                label = 'Normal' if pred == 1 else 'Abnormal'
                ax2.plot(signal, color=color, linewidth=2, label=f'Kelas: {label}')
                ax2.set_title(f"Prediksi: **{label}**", fontsize=12, color=color)
                ax2.set_xlabel("Waktu (96 titik)")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)

                if pred == 1:
                    st.markdown("🟢 **Interpretasi**: Sinyal ECG menunjukkan pola **normal**.")
                else:
                    st.markdown("🔴 **Interpretasi**: Sinyal ECG menunjukkan pola **abnormal** yang konsisten dengan **Infark Miokard (Serangan Jantung)**.")
                    st.markdown("""
                    **Apa itu Infark Miokard?**  
                    Kondisi darurat medis akibat aliran darah ke jantung terhambat. Pada ECG, sering muncul:
                    - Perubahan segmen ST,
                    - Gelombang T terbalik,
                    - Gelombang Q abnormal.

                    **Gejala umum**: nyeri dada berat, sesak napas, keringat dingin.  
                    ⚠️ Segera hubungi tenaga medis jika mengalami gejala ini.
                    """)

                st.info("ℹ️ **Peringatan**: Aplikasi ini hanya untuk keperluan edukasi dan eksperimen akademik — **bukan pengganti diagnosis dokter**.")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")