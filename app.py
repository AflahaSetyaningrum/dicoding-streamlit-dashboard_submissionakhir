import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# =========================
# CONFIG PAGE
# =========================
st.set_page_config(
    page_title="Dashboard Dropout Siswa",
    layout="wide"
)

# =========================
# LOAD DATA (ANTI ERROR PATH)
# =========================
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)

    # coba beberapa kemungkinan nama file
    possible_files = [
        os.path.join(base_path, "data", "students_performance.csv"),
        os.path.join(base_path, "data", "students.csv"),
        os.path.join(base_path, "students_performance.csv"),
        os.path.join(base_path, "students.csv"),
    ]

    for file in possible_files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            return df

    st.error("Dataset tidak ditemukan. Pastikan file CSV ada di folder project.")
    st.stop()

df = load_data()

# =========================
# DEBUG KOLOM (biar aman)
# =========================
st.sidebar.write("Kolom Dataset:")
st.sidebar.write(df.columns.tolist())

# =========================
# DETEKSI KOLOM SCORE OTOMATIS
# =========================
score_cols = [col for col in df.columns if "score" in col.lower()]

if len(score_cols) >= 3:
    df["average_score"] = df[score_cols].mean(axis=1)
else:
    st.error("Kolom score tidak ditemukan di dataset!")
    st.stop()

# =========================
# BUAT LABEL DROPOUT (JIKA BELUM ADA)
# =========================
if "dropout" not in df.columns:
    df["dropout"] = np.where(df["average_score"] < 60, 1, 0)

# =========================
# SIDEBAR MENU
# =========================
st.sidebar.title("Menu")
menu = st.sidebar.radio("Pilih Halaman", ["Dashboard", "Prediksi"])

# =========================
# DASHBOARD
# =========================
if menu == "Dashboard":
    st.title("📊 Dashboard Analisis Dropout Siswa")

    # METRICS
    col1, col2, col3 = st.columns(3)

    total = len(df)
    dropout_total = df["dropout"].sum()
    dropout_rate = (dropout_total / total) * 100

    col1.metric("Total Siswa", total)
    col2.metric("Total Dropout", int(dropout_total))
    col3.metric("Dropout Rate (%)", f"{dropout_rate:.2f}")

    st.markdown("---")

    # DISTRIBUSI DROPOUT
    st.subheader("Distribusi Dropout")

    counts = df["dropout"].value_counts()

    fig1, ax1 = plt.subplots()
    ax1.bar(["Tidak Dropout", "Dropout"], counts.values)
    st.pyplot(fig1)

    # FILTER (jika ada gender)
    if "gender" in df.columns:
        st.subheader("Filter")

        gender = st.selectbox("Pilih Gender", ["All"] + list(df["gender"].unique()))

        if gender != "All":
            df_filtered = df[df["gender"] == gender]
        else:
            df_filtered = df
    else:
        df_filtered = df

    # SCATTER NILAI
    st.subheader("Analisis Nilai vs Dropout")

    fig2, ax2 = plt.subplots()
    ax2.scatter(df_filtered[score_cols[0]], df_filtered["dropout"])
    ax2.set_xlabel(score_cols[0])
    ax2.set_ylabel("Dropout")
    st.pyplot(fig2)

    # RATA-RATA NILAI
    st.subheader("Rata-rata Nilai")

    avg = df.groupby("dropout")[score_cols].mean()

    fig3, ax3 = plt.subplots()
    avg.plot(kind="bar", ax=ax3)
    st.pyplot(fig3)

    # FEATURE IMPORTANCE (dummy)
    st.subheader("Feature Importance")

    importance = np.random.rand(len(score_cols))

    fig4, ax4 = plt.subplots()
    ax4.bar(score_cols, importance)
    st.pyplot(fig4)

# =========================
# PREDIKSI
# =========================
else:
    st.title("🤖 Prediksi Dropout")

    st.write("Masukkan nilai siswa:")

    # INPUT DINAMIS
    inputs = []
    for col in score_cols[:3]:
        val = st.slider(col, 0, 100, 50)
        inputs.append(val)

    # LOAD MODEL (optional)
    model = None
    model_path = os.path.join(os.path.dirname(__file__), "model", "model.pkl")

    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    # PREDIKSI
    if st.button("Prediksi"):
        data = np.array([inputs])

        if model:
            pred = model.predict(data)[0]
        else:
            # fallback rule
            avg = np.mean(data)
            pred = 1 if avg < 60 else 0

        if pred == 1:
            st.error("⚠️ Berpotensi Dropout")
        else:
            st.success("✅ Tidak Dropout")