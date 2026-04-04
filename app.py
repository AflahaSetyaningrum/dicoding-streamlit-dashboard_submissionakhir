import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================
# CONFIG
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
    return pd.read_csv("data/students.csv")

df = load_data()

# =========================
# LOAD MODEL BUNDLE
# =========================
@st.cache_resource
def load_model():
    return joblib.load("model/model.pkl")

bundle = load_model()

model = bundle["model"]
scaler = bundle["scaler"]
label_encoders = bundle["label_encoders"]
features = bundle["features"]

# =========================
# SIDEBAR FILTER
# =========================
st.sidebar.header("Filter Data")

if 'dropout' in df.columns:
    selected = st.sidebar.multiselect(
        "Status Dropout",
        df['dropout'].unique(),
        default=df['dropout'].unique()
    )
    df_filtered = df[df['dropout'].isin(selected)]
else:
    df_filtered = df.copy()

# =========================
# TITLE
# =========================
st.title("📊 Dashboard Monitoring Dropout Siswa")
st.write("Jaya Jaya Institut")

# =========================
# METRICS
# =========================
col1, col2 = st.columns(2)

if 'dropout' in df.columns:
    col1.metric("Total Siswa", len(df_filtered))
    col2.metric("Total Dropout", int(df_filtered['dropout'].sum()))

# =========================
# DISTRIBUSI DROPOUT
# =========================
st.subheader("Distribusi Dropout")

if 'dropout' in df.columns:
    fig, ax = plt.subplots()
    df_filtered['dropout'].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)

# =========================
# ANALISIS FITUR
# =========================
st.subheader("Analisis Fitur")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

selected_feature = st.selectbox("Pilih Fitur", numeric_cols)

if 'dropout' in df.columns:
    fig, ax = plt.subplots()
    ax.boxplot([
        df_filtered[df_filtered['dropout'] == 0][selected_feature],
        df_filtered[df_filtered['dropout'] == 1][selected_feature]
    ])
    ax.set_xticklabels(["Tidak", "Dropout"])
    st.pyplot(fig)

# =========================
# FEATURE IMPORTANCE
# =========================
st.subheader("Feature Importance")

try:
    importances = model.feature_importances_

    fi_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    fig, ax = plt.subplots()
    ax.barh(fi_df['Feature'][:10], fi_df['Importance'][:10])
    ax.invert_yaxis()
    st.pyplot(fig)

except:
    st.warning("Feature importance tidak tersedia")

# =========================
# PREDIKSI
# =========================
st.subheader("🎯 Prediksi Dropout")

input_data = {}

for col in features:
    if col in label_encoders:
        input_data[col] = st.selectbox(col, label_encoders[col].classes_)
    else:
        input_data[col] = st.number_input(col, value=0.0)

input_df = pd.DataFrame([input_data])

# =========================
# PREPROCESS INPUT
# =========================
for col in input_df.columns:
    if col in label_encoders:
        le = label_encoders[col]
        input_df[col] = le.transform(input_df[col])

input_df = input_df[features]

input_scaled = scaler.transform(input_df)

# =========================
# PREDIKSI BUTTON
# =========================
if st.button("Prediksi"):
    result = model.predict(input_scaled)

    if result[0] == 1:
        st.error("⚠️ Berpotensi Dropout")
    else:
        st.success("✅ Tidak Dropout")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Proyek Akhir Dicoding")