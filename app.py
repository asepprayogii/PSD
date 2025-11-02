import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import time

# ===============================
# Konfigurasi Halaman
# ===============================
st.set_page_config(
    page_title="Prediksi NO₂ Kota Medan",
    page_icon="🌫️",
    layout="wide"
)

st.title("🌫️ Prediksi Kadar NO₂ Kota Medan Berbasis KNN")
st.markdown(
    "Masukkan kadar NO₂ 5 hari terakhir untuk memprediksi kadar NO₂ besok (t+1), "
    "2 hari lagi (t+2), atau keduanya menggunakan model K-Nearest Neighbors."
)
st.markdown("---")

# ===============================
# Path Model
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

# ===============================
# Load Semua Model
# ===============================
@st.cache_resource
def load_all_models():
    try:
        models = {}
        scalers = {}
        for i in range(1, 6):
            models[f"day{i}"] = joblib.load(os.path.join(MODEL_DIR, f"knn_model_day{i}.pkl"))
            scalers[f"day{i}"] = joblib.load(os.path.join(MODEL_DIR, f"scaler_day{i}.pkl"))
        return models, scalers, True, "✅ Semua model berhasil dimuat!"
    except Exception as e:
        return None, None, False, f"❌ Error memuat model: {e}"

models, scalers, loaded, load_message = load_all_models()
st.info(load_message)

# ===============================
# Input 5 Hari NO2
# ===============================
st.subheader("1) Input NO₂ 5 Hari Terakhir")

cols = st.columns(5)
labels = ["Kemarin (t-1)", "-2 Hari (t-2)", "-3 Hari (t-3)", "-4 Hari (t-4)", "-5 Hari (t-5)"]

inputs = []
for i in range(5):
    val = cols[i].number_input(
        labels[i],
        min_value=0.0,
        max_value=1.0,
        value=0.000000,
        format="%.8f"
    )
    inputs.append(val)

# ===============================
# Pilih Mode Prediksi
# ===============================
st.subheader("2) Pilih Mode Prediksi")

mode = st.radio(
    "Jenis prediksi:",
    ["Besok (t+1)", "2 Hari Lagi (t+2)", "Keduanya"],
    horizontal=True
)

# ===============================
# Input Threshold
# ===============================
st.subheader("3) Threshold Kategori")

THRESHOLD = st.number_input(
    "Threshold NO₂ (mol/m²)",
    min_value=0.0,
    max_value=1.0,
    value=0.000050,
    format="%.8f",
    help="Jika prediksi ≤ threshold → Baik, jika > threshold → Buruk"
)

# ===============================
# Tombol Prediksi
# ===============================
predict = st.button("🔮 Jalankan Prediksi", type="primary", use_container_width=True)

# ===============================
# Proses Prediksi
# ===============================
if predict:

    if not loaded:
        st.error("Model belum berhasil dimuat!")
        st.stop()

    with st.spinner("Sedang melakukan prediksi... 🔄"):
        time.sleep(1)

        df_input = pd.DataFrame([inputs], columns=["t-1", "t-2", "t-3", "t-4", "t-5"])
        results = {}

        # ===== Prediksi T+1 =====
        if mode in ["Besok (t+1)", "Keduanya"]:
            X1 = df_input[["t-1"]]
            X1_scaled = scalers["day1"].transform(X1)
            results["t+1"] = models["day1"].predict(X1_scaled)[0]

        # ===== Prediksi T+2 =====
        if mode in ["2 Hari Lagi (t+2)", "Keduanya"]:
            X2 = df_input[["t-1", "t-2"]]
            X2_scaled = scalers["day2"].transform(X2)
            results["t+2"] = models["day2"].predict(X2_scaled)[0]

    # ===============================
    # Hasil Prediksi
    # ===============================
    st.subheader("📈 Hasil Prediksi")

    for key, val in results.items():

        status = "✅ Baik" if val <= THRESHOLD else "⚠️ Buruk"
        color = "green" if val <= THRESHOLD else "red"

        st.markdown(f"### Prediksi {key}")

        colA, colB = st.columns(2)
        colA.metric(
            label=f"Hasil Prediksi {key}",
            value=f"{val:.8f}",
            delta=f"{val - THRESHOLD:.8f}"
        )
        colB.metric(
            label="Status",
            value=status,
            delta=f"{((val - THRESHOLD) / THRESHOLD) * 100:+.2f}%"
        )

        # ===============================
        # Grafik
        # ===============================
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["Rata-rata 5 Hari", f"Prediksi {key}"],
               [np.mean(inputs), val],
               color=["skyblue", color])

        ax.axhline(THRESHOLD, color="orange", linestyle="--", label=f"Threshold {THRESHOLD}")
        ax.set_ylabel("Kadar NO₂ (mol/m²)")
        ax.legend()
        st.pyplot(fig)

# ===============================
# Detail Input
# ===============================
if predict:
    with st.expander("📋 Detail Input"):
        st.write(df_input)

# ===============================
# Footer
# ===============================
st.markdown("---")
st.markdown("<center>Developed with ❤️ | Prediksi NO₂ Kota Medan</center>", unsafe_allow_html=True)
