import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# =========================
# CONFIG PAGE
# =========================
st.set_page_config(
    page_title="Dashboard Dropout Siswa",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("data/students_performance.csv")
    return df

df = load_data()

# =========================
# PREPROCESS (SIMPLE)
# =========================
# Contoh membuat label dropout sederhana (jika belum ada)
if "dropout" not in df.columns:
    df["average_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)
    df["dropout"] = np.where(df["average_score"] < 60, 1, 0)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Menu")
menu = st.sidebar.radio(
    "Pilih Halaman",
    ["Dashboard", "Prediksi Dropout"]
)

# =========================
# DASHBOARD
# =========================
if menu == "Dashboard":
    st.title("📊 Dashboard Analisis Dropout Siswa")

    # =========================
    # METRICS
    # =========================
    col1, col2, col3 = st.columns(3)

    total_siswa = len(df)
    total_dropout = df["dropout"].sum()
    dropout_rate = (total_dropout / total_siswa) * 100

    col1.metric("Total Siswa", total_siswa)
    col2.metric("Total Dropout", total_dropout)
    col3.metric("Dropout Rate (%)", f"{dropout_rate:.2f}")

    st.markdown("---")

    # =========================
    # DISTRIBUSI DROPOUT
    # =========================
    st.subheader("Distribusi Dropout Siswa")

    dropout_counts = df["dropout"].value_counts()

    fig1, ax1 = plt.subplots()
    ax1.bar(["Tidak Dropout", "Dropout"], dropout_counts.values)
    st.pyplot(fig1)

    # =========================
    # FILTER
    # =========================
    st.subheader("Filter Data")

    gender_filter = st.selectbox("Pilih Gender", ["All"] + list(df["gender"].unique()))

    df_filtered = df.copy()
    if gender_filter != "All":
        df_filtered = df_filtered[df_filtered["gender"] == gender_filter]

    # =========================
    # ANALISIS NILAI
    # =========================
    st.subheader("Analisis Nilai vs Dropout")

    fig2, ax2 = plt.subplots()
    ax2.scatter(df_filtered["math score"], df_filtered["dropout"])
    ax2.set_xlabel("Math Score")
    ax2.set_ylabel("Dropout")
    st.pyplot(fig2)

    # =========================
    # RATA-RATA NILAI
    # =========================
    st.subheader("Rata-rata Nilai Berdasarkan Status")

    avg_scores = df.groupby("dropout")[["math score", "reading score", "writing score"]].mean()

    fig3, ax3 = plt.subplots()
    avg_scores.plot(kind="bar", ax=ax3)
    st.pyplot(fig3)

    # =========================
    # FEATURE IMPORTANCE (DUMMY)
    # =========================
    st.subheader("Feature Importance (Simulasi)")

    features = ["math score", "reading score", "writing score"]
    importance = [0.4, 0.3, 0.3]

    fig4, ax4 = plt.subplots()
    ax4.bar(features, importance)
    st.pyplot(fig4)


# =========================
# PREDIKSI
# =========================
elif menu == "Prediksi Dropout":
    st.title("🤖 Prediksi Dropout Siswa")

    st.write("Masukkan data siswa:")

    # =========================
    # INPUT USER
    # =========================
    gender = st.selectbox("Gender", ["male", "female"])
    math_score = st.slider("Math Score", 0, 100, 50)
    reading_score = st.slider("Reading Score", 0, 100, 50)
    writing_score = st.slider("Writing Score", 0, 100, 50)

    # =========================
    # LOAD MODEL (OPTIONAL)
    # =========================
    model = None
    if os.path.exists("model/model.pkl"):
        with open("model/model.pkl", "rb") as f:
            model = pickle.load(f)

    # =========================
    # PREDIKSI
    # =========================
    if st.button("Prediksi"):
        input_data = np.array([[math_score, reading_score, writing_score]])

        if model:
            prediction = model.predict(input_data)[0]
        else:
            # fallback rule-based
            avg = np.mean(input_data)
            prediction = 1 if avg < 60 else 0

        if prediction == 1:
            st.error("⚠️ Siswa Berpotensi Dropout")
        else:
            st.success("✅ Siswa Tidak Dropout")
