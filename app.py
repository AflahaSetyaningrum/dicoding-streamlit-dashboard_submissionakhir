import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Dashboard Dropout Siswa",
    layout="wide"
)

# =========================
# LOAD DATA (ANTI ERROR)
# =========================
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)

    possible_files = [
        os.path.join(base_path, "data", "students.csv"),
        os.path.join(base_path, "data", "students_performance.csv"),
        os.path.join(base_path, "students.csv"),
        os.path.join(base_path, "students_performance.csv"),
    ]

    for file in possible_files:
        if os.path.exists(file):
            try:
                df = pd.read_csv(file)
            except:
                df = pd.read_csv(file, delimiter=";")  # fallback delimiter

            return df

    st.error("❌ Dataset tidak ditemukan")
    st.stop()

df = load_data()

# =========================
# DEBUG (WAJIB UNTUK CEK)
# =========================
st.sidebar.write("📌 Kolom Dataset:")
st.sidebar.write(df.columns.tolist())

# =========================
# KONVERSI NUMERIK (FIX PANDAS)
# =========================
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# hapus NaN
df = df.dropna()

# =========================
# AMBIL KOLOM NUMERIK
# =========================
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# buang kolom id
numeric_cols = [col for col in numeric_cols if "id" not in col.lower()]

# fallback jika masih kosong
if len(numeric_cols) < 2:
    st.error("❌ Dataset tidak memiliki cukup kolom numerik")
    st.stop()

# pilih 3 kolom pertama
score_cols = numeric_cols[:3]

# =========================
# FEATURE ENGINEERING
# =========================
df["average_score"] = df[score_cols].mean(axis=1)

# =========================
# TARGET
# =========================
if "Status" in df.columns:
    df["dropout"] = df["Status"].apply(lambda x: 1 if x == "Dropout" else 0)
else:
    df["dropout"] = np.where(
        df["average_score"] < df["average_score"].mean(), 1, 0
    )

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Menu")
menu = st.sidebar.radio("Pilih Menu", ["Dashboard", "Prediksi"])

# =========================
# DASHBOARD
# =========================
if menu == "Dashboard":
    st.title("📊 Dashboard Dropout Siswa")

    col1, col2, col3 = st.columns(3)

    total = len(df)
    dropout_total = df["dropout"].sum()
    rate = (dropout_total / total) * 100

    col1.metric("Total Siswa", total)
    col2.metric("Total Dropout", int(dropout_total))
    col3.metric("Dropout Rate (%)", f"{rate:.2f}")

    st.markdown("---")

    # DISTRIBUSI
    st.subheader("Distribusi Dropout")

    fig1, ax1 = plt.subplots()
    ax1.bar(["Tidak Dropout", "Dropout"], df["dropout"].value_counts().values)
    st.pyplot(fig1)

    # SCATTER
    st.subheader("Analisis Nilai vs Dropout")

    fig2, ax2 = plt.subplots()
    ax2.scatter(df[score_cols[0]], df["dropout"])
    ax2.set_xlabel(score_cols[0])
    ax2.set_ylabel("Dropout")
    st.pyplot(fig2)

    # RATA-RATA
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

    st.write("Masukkan data:")

    inputs = []
    for col in score_cols:
        val = st.slider(
            col,
            float(df[col].min()),
            float(df[col].max()),
            float(df[col].mean())
        )
        inputs.append(val)

    if st.button("Prediksi"):
        avg = np.mean(inputs)
        pred = 1 if avg < df["average_score"].mean() else 0

        if pred == 1:
            st.error("⚠️ Berpotensi Dropout")
        else:
            st.success("✅ Tidak Dropout")