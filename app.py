import pickle
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Jaya Jaya Institut – Student Dropout Monitor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSlider label {
    color: #94a3b8 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Main background */
.main { background-color: #f8fafc; }

/* Metric cards */
.metric-card {
    background: white;
    border-radius: 16px;
    padding: 24px 28px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07), 0 4px 16px rgba(0,0,0,0.04);
    border-left: 4px solid;
    height: 100%;
}
.metric-card.red   { border-color: #ef4444; }
.metric-card.green { border-color: #22c55e; }
.metric-card.blue  { border-color: #3b82f6; }
.metric-card.amber { border-color: #f59e0b; }

.metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #94a3b8;
    font-family: 'DM Sans', sans-serif;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1;
    color: #0f172a;
}
.metric-sub {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 4px;
}

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #0f172a;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 8px;
    margin-bottom: 16px;
}

/* Prediction result boxes */
.pred-dropout {
    background: linear-gradient(135deg, #fef2f2, #fee2e2);
    border: 2px solid #ef4444;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}
.pred-graduate {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 2px solid #22c55e;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}
.pred-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
}
.pred-sub {
    font-size: 0.9rem;
    color: #475569;
    margin-top: 8px;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #f1f5f9;
    padding: 4px;
    border-radius: 12px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    color: #64748b;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #0f172a !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  Load model assets
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('model/feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    return model, scaler, feature_columns

@st.cache_data
def load_data():
    df = pd.read_csv('data/students.csv', sep=';')
    df.columns = df.columns.str.strip()
    return df

model, scaler, feature_columns = load_model()
df = load_data()

# ─────────────────────────────────────────
#  Sidebar – Navigation
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 16px 0 24px 0;'>
        <div style='font-family:Syne,sans-serif; font-size:1.4rem; font-weight:800; color:#f8fafc;'>🎓 JJI Monitor</div>
        <div style='font-size:0.75rem; color:#64748b; margin-top:4px;'>Jaya Jaya Institut</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigasi",
        ["📊 Dashboard", "🔮 Prediksi Siswa"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("<div style='font-size:0.7rem; color:#475569;'>Dataset Overview</div>", unsafe_allow_html=True)

    total = len(df)
    dropout_n = (df['Status'] == 'Dropout').sum()
    graduate_n = (df['Status'] == 'Graduate').sum()
    enrolled_n = (df['Status'] == 'Enrolled').sum()

    st.markdown(f"<div style='font-size:0.82rem; color:#94a3b8; margin-top:8px;'>Total siswa: <b style='color:#e2e8f0'>{total:,}</b></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.82rem; color:#ef4444;'>Dropout: <b>{dropout_n:,}</b></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.82rem; color:#22c55e;'>Graduate: <b>{graduate_n:,}</b></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.82rem; color:#3b82f6;'>Enrolled: <b>{enrolled_n:,}</b></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  COLOR PALETTE
# ─────────────────────────────────────────
C_DROPOUT  = "#ef4444"
C_GRADUATE = "#22c55e"
C_ENROLLED = "#3b82f6"
C_BG       = "#f8fafc"

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8fafc',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.grid': True,
    'grid.color': '#e2e8f0',
    'grid.linewidth': 0.6,
    'font.family': 'DejaVu Sans',
})

# ═════════════════════════════════════════
#  PAGE 1: DASHBOARD
# ═════════════════════════════════════════
if page == "📊 Dashboard":

    st.markdown("<h1 style='font-family:Syne,sans-serif; font-size:2rem; font-weight:800; color:#0f172a; margin-bottom:4px;'>Student Dropout Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; margin-bottom:28px;'>Monitor performa dan risiko dropout siswa Jaya Jaya Institut</p>", unsafe_allow_html=True)

    # ── KPI Cards ──
    dropout_rate = dropout_n / (dropout_n + graduate_n) * 100
    avg_age      = df['Age_at_enrollment'].mean()
    scholar_drop = df[df['Status']=='Dropout']['Scholarship_holder'].mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card red">
            <div class="metric-label">Dropout Rate</div>
            <div class="metric-value" style="color:#ef4444">{dropout_rate:.1f}%</div>
            <div class="metric-sub">{dropout_n:,} dari {dropout_n+graduate_n:,} siswa</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card green">
            <div class="metric-label">Graduate</div>
            <div class="metric-value" style="color:#22c55e">{graduate_n:,}</div>
            <div class="metric-sub">{graduate_n/(dropout_n+graduate_n)*100:.1f}% tingkat kelulusan</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card blue">
            <div class="metric-label">Masih Enrolled</div>
            <div class="metric-value" style="color:#3b82f6">{enrolled_n:,}</div>
            <div class="metric-sub">Siswa aktif saat ini</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card amber">
            <div class="metric-label">Rata-rata Usia</div>
            <div class="metric-value" style="color:#f59e0b">{avg_age:.1f}</div>
            <div class="metric-sub">tahun saat pendaftaran</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──
    tab1, tab2, tab3 = st.tabs(["  📈 Distribusi & Demografi  ", "  📚 Performa Akademik  ", "  🏆 Faktor Risiko  "])

    # ── TAB 1 ──
    with tab1:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="section-header">Distribusi Status Siswa</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            sizes  = [dropout_n, graduate_n, enrolled_n]
            colors = [C_DROPOUT, C_GRADUATE, C_ENROLLED]
            labels = ['Dropout', 'Graduate', 'Enrolled']
            wedges, texts, autotexts = ax.pie(
                sizes, labels=None, colors=colors,
                autopct='%1.1f%%', startangle=90,
                wedgeprops=dict(width=0.55, edgecolor='white', linewidth=3),
                pctdistance=0.75
            )
            for at in autotexts:
                at.set_fontsize(11)
                at.set_fontweight('bold')
                at.set_color('white')
            ax.legend(wedges, labels, loc='lower center', ncol=3,
                      bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=10)
            ax.set_facecolor('white')
            fig.patch.set_facecolor('white')
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_b:
            st.markdown('<div class="section-header">Status Siswa berdasarkan Gender</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 4))
            gender_status = df.groupby(['Gender', 'Status']).size().unstack(fill_value=0)
            gender_status.index = ['Perempuan' if g == 0 else 'Laki-laki' for g in gender_status.index]
            cols_order = [c for c in ['Dropout', 'Graduate', 'Enrolled'] if c in gender_status.columns]
            gender_status[cols_order].plot(
                kind='bar', ax=ax,
                color=[C_DROPOUT, C_GRADUATE, C_ENROLLED][:len(cols_order)],
                edgecolor='white', linewidth=1.5, width=0.6
            )
            ax.set_xlabel('')
            ax.set_ylabel('Jumlah Siswa', fontsize=9, color='#64748b')
            ax.tick_params(axis='x', rotation=0)
            ax.legend(frameon=False, fontsize=9)
            ax.spines['bottom'].set_color('#e2e8f0')
            st.pyplot(fig, use_container_width=True)
            plt.close()

        col_c, col_d = st.columns(2)

        with col_c:
            st.markdown('<div class="section-header">Distribusi Usia saat Pendaftaran</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 4))
            for status, color in [('Dropout', C_DROPOUT), ('Graduate', C_GRADUATE)]:
                data = df[df['Status'] == status]['Age_at_enrollment']
                ax.hist(data, bins=20, alpha=0.65, color=color, label=status, edgecolor='white')
            ax.set_xlabel('Usia', fontsize=9, color='#64748b')
            ax.set_ylabel('Frekuensi', fontsize=9, color='#64748b')
            ax.legend(frameon=False, fontsize=9)
            ax.spines['bottom'].set_color('#e2e8f0')
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_d:
            st.markdown('<div class="section-header">Status vs Pemegang Beasiswa</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 4))
            schol = df[df['Status'].isin(['Dropout','Graduate'])].groupby(['Scholarship_holder','Status']).size().unstack(fill_value=0)
            schol.index = ['Tidak' if i == 0 else 'Ya' for i in schol.index]
            pct = schol.div(schol.sum(axis=1), axis=0) * 100
            cols_o = [c for c in ['Dropout','Graduate'] if c in pct.columns]
            pct[cols_o].plot(kind='bar', ax=ax,
                             color=[C_DROPOUT, C_GRADUATE][:len(cols_o)],
                             edgecolor='white', linewidth=1.5, width=0.5)
            ax.set_xlabel('Pemegang Beasiswa', fontsize=9, color='#64748b')
            ax.set_ylabel('Persentase (%)', fontsize=9, color='#64748b')
            ax.tick_params(axis='x', rotation=0)
            ax.legend(frameon=False, fontsize=9)
            ax.spines['bottom'].set_color('#e2e8f0')
            st.pyplot(fig, use_container_width=True)
            plt.close()

    # ── TAB 2 ──
    with tab2:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="section-header">Nilai Semester 1 vs Status</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 4))
            for status, color in [('Dropout', C_DROPOUT), ('Graduate', C_GRADUATE)]:
                data = df[df['Status'] == status]['Curricular_units_1st_sem_grade']
                ax.hist(data, bins=25, alpha=0.65, color=color, label=status, edgecolor='white')
            ax.set_xlabel('Nilai Semester 1', fontsize=9, color='#64748b')
            ax.set_ylabel('Frekuensi', fontsize=9, color='#64748b')
            ax.legend(frameon=False, fontsize=9)
            ax.spines['bottom'].set_color('#e2e8f0')
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_b:
            st.markdown('<div class="section-header">Nilai Semester 2 vs Status</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 4))
            for status, color in [('Dropout', C_DROPOUT), ('Graduate', C_GRADUATE)]:
                data = df[df['Status'] == status]['Curricular_units_2nd_sem_grade']
                ax.hist(data, bins=25, alpha=0.65, color=color, label=status, edgecolor='white')
            ax.set_xlabel('Nilai Semester 2', fontsize=9, color='#64748b')
            ax.set_ylabel('Frekuensi', fontsize=9, color='#64748b')
            ax.legend(frameon=False, fontsize=9)
            ax.spines['bottom'].set_color('#e2e8f0')
            st.pyplot(fig, use_container_width=True)
            plt.close()

        col_c, col_d = st.columns(2)

        with col_c:
            st.markdown('<div class="section-header">Unit Kurikulum Disetujui (Sem 1)</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 4))
            plot_df = df[df['Status'].isin(['Dropout','Graduate'])]
            sns.boxplot(data=plot_df, x='Status', y='Curricular_units_1st_sem_approved',
                        palette={'Dropout': C_DROPOUT, 'Graduate': C_GRADUATE}, ax=ax,
                        width=0.5, linewidth=1.5, fliersize=3)
            ax.set_xlabel('')
            ax.set_ylabel('Unit Disetujui', fontsize=9, color='#64748b')
            ax.spines['bottom'].set_color('#e2e8f0')
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_d:
            st.markdown('<div class="section-header">Jumlah Mata Kuliah Terdaftar</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 4))
            plot_df = df[df['Status'].isin(['Dropout','Graduate'])]
            sns.boxplot(data=plot_df, x='Status', y='Curricular_units_1st_sem_enrolled',
                        palette={'Dropout': C_DROPOUT, 'Graduate': C_GRADUATE}, ax=ax,
                        width=0.5, linewidth=1.5, fliersize=3)
            ax.set_xlabel('')
            ax.set_ylabel('Jumlah Terdaftar', fontsize=9, color='#64748b')
            ax.spines['bottom'].set_color('#e2e8f0')
            st.pyplot(fig, use_container_width=True)
            plt.close()

    # ── TAB 3 ──
    with tab3:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="section-header">Status vs Debitur (Tunggakan)</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 4))
            debt = df[df['Status'].isin(['Dropout','Graduate'])].groupby(['Debtor','Status']).size().unstack(fill_value=0)
            debt.index = ['Tidak' if i == 0 else 'Ya' for i in debt.index]
            pct = debt.div(debt.sum(axis=1), axis=0) * 100
            cols_o = [c for c in ['Dropout','Graduate'] if c in pct.columns]
            pct[cols_o].plot(kind='bar', ax=ax,
                             color=[C_DROPOUT, C_GRADUATE][:len(cols_o)],
                             edgecolor='white', linewidth=1.5, width=0.5)
            ax.set_xlabel('Memiliki Tunggakan', fontsize=9, color='#64748b')
            ax.set_ylabel('Persentase (%)', fontsize=9, color='#64748b')
            ax.tick_params(axis='x', rotation=0)
            ax.legend(frameon=False, fontsize=9)
            ax.spines['bottom'].set_color('#e2e8f0')
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_b:
            st.markdown('<div class="section-header">Status vs Biaya Kuliah Terkini</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 4))
            tuition = df[df['Status'].isin(['Dropout','Graduate'])].groupby(['Tuition_fees_up_to_date','Status']).size().unstack(fill_value=0)
            tuition.index = ['Tidak' if i == 0 else 'Ya' for i in tuition.index]
            pct2 = tuition.div(tuition.sum(axis=1), axis=0) * 100
            cols_o2 = [c for c in ['Dropout','Graduate'] if c in pct2.columns]
            pct2[cols_o2].plot(kind='bar', ax=ax,
                               color=[C_DROPOUT, C_GRADUATE][:len(cols_o2)],
                               edgecolor='white', linewidth=1.5, width=0.5)
            ax.set_xlabel('Biaya Terkini', fontsize=9, color='#64748b')
            ax.set_ylabel('Persentase (%)', fontsize=9, color='#64748b')
            ax.tick_params(axis='x', rotation=0)
            ax.legend(frameon=False, fontsize=9)
            ax.spines['bottom'].set_color('#e2e8f0')
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # Feature Importance
        importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values()
        fig, ax = plt.subplots(figsize=(9, 5))
        colors_bar = ['#3b82f6' if i >= 5 else C_DROPOUT for i in range(len(importances))]
        bars = ax.barh(importances.index[::-1], importances.values[::-1],
                    color=colors_bar[::-1], edgecolor='white', linewidth=1)
        ax.set_xlabel('Importance Score', fontsize=9, color='#64748b')
        ax.spines['bottom'].set_color('#e2e8f0')
        for bar, val in zip(bars, importances.values[::-1]):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=8, color='#475569')
        legend_patches = [
            mpatches.Patch(color=C_DROPOUT, label='Top 5'),
            mpatches.Patch(color='#3b82f6', label='Bottom 5'),
        ]
        ax.legend(handles=legend_patches, frameon=False, fontsize=8, loc='lower right')

        st.pyplot(fig, use_container_width=True)
        plt.close()


# ═════════════════════════════════════════
#  PAGE 2: PREDIKSI
# ═════════════════════════════════════════
else:
    st.markdown("<h1 style='font-family:Syne,sans-serif; font-size:2rem; font-weight:800; color:#0f172a; margin-bottom:4px;'>🔮 Prediksi Risiko Dropout</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; margin-bottom:24px;'>Masukkan data siswa untuk memprediksi apakah siswa berisiko Dropout atau Graduate</p>", unsafe_allow_html=True)

    col_form, col_result = st.columns([1.2, 1])

    with col_form:
        st.markdown('<div class="section-header">Data Akademik & Pribadi Siswa</div>', unsafe_allow_html=True)

        with st.container():
            c1, c2 = st.columns(2)
            with c1:
                age        = st.number_input("Usia saat Pendaftaran", 17, 60, 20)
                gender     = st.selectbox("Gender", [0, 1], format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki")
                scholarship = st.selectbox("Pemegang Beasiswa", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
                debtor     = st.selectbox("Memiliki Tunggakan", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
                tuition    = st.selectbox("Biaya Kuliah Terkini", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
            with c2:
                grade_1    = st.slider("Nilai Semester 1", 0.0, 20.0, 12.0, 0.1)
                grade_2    = st.slider("Nilai Semester 2", 0.0, 20.0, 12.0, 0.1)
                approved_1 = st.number_input("Unit Disetujui Sem 1", 0, 30, 5)
                approved_2 = st.number_input("Unit Disetujui Sem 2", 0, 30, 5)
                enrolled_1 = st.number_input("Unit Terdaftar Sem 1", 0, 30, 6)

        predict_btn = st.button("🔍 Prediksi Sekarang", use_container_width=True, type="primary")

    with col_result:
        st.markdown('<div class="section-header">Hasil Prediksi</div>', unsafe_allow_html=True)

        if predict_btn:
            # Build input dengan semua feature_columns
            input_dict = {
                'Age_at_enrollment': age,
                'Gender': gender,
                'Scholarship_holder': scholarship,
                'Debtor': debtor,
                'Tuition_fees_up_to_date': tuition,
                'Curricular_units_1st_sem_grade': grade_1,
                'Curricular_units_2nd_sem_grade': grade_2,
                'Curricular_units_1st_sem_approved': approved_1,
                'Curricular_units_2nd_sem_approved': approved_2,
                'Curricular_units_1st_sem_enrolled': enrolled_1,
    }

            input_df = pd.DataFrame([input_dict])[feature_columns]
            input_scaled = scaler.transform(input_df)

            pred       = model.predict(input_scaled)[0]
            pred_proba = model.predict_proba(input_scaled)[0]

            dropout_prob  = pred_proba[0] * 100
            graduate_prob = pred_proba[1] * 100

            if pred == 0:
                st.markdown(f"""
                <div class="pred-dropout">
                    <div class="pred-title" style="color:#dc2626;">⚠️ BERISIKO DROPOUT</div>
                    <div style="font-size:2.5rem; font-weight:800; color:#ef4444; margin:12px 0;">{dropout_prob:.1f}%</div>
                    <div class="pred-sub">Probabilitas siswa ini akan Dropout</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-graduate">
                    <div class="pred-title" style="color:#16a34a;">✅ DIPREDIKSI LULUS</div>
                    <div style="font-size:2.5rem; font-weight:800; color:#22c55e; margin:12px 0;">{graduate_prob:.1f}%</div>
                    <div class="pred-sub">Probabilitas siswa ini akan Graduate</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Probability bar chart
            fig, ax = plt.subplots(figsize=(5, 2.2))
            bars = ax.barh(['Graduate', 'Dropout'],
                           [graduate_prob, dropout_prob],
                           color=[C_GRADUATE, C_DROPOUT],
                           edgecolor='white', height=0.45)
            for bar, val in zip(bars, [graduate_prob, dropout_prob]):
                ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                        f'{val:.1f}%', va='center', fontsize=10, fontweight='bold', color='#0f172a')
            ax.set_xlim(0, 115)
            ax.set_xlabel('Probabilitas (%)', fontsize=9, color='#64748b')
            ax.spines['bottom'].set_color('#e2e8f0')
            ax.set_facecolor('white')
            fig.patch.set_facecolor('white')
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # Recommendation
            st.markdown("<br>", unsafe_allow_html=True)
            if pred == 0:
                st.error("**Rekomendasi:** Siswa ini memerlukan perhatian khusus. Pertimbangkan program bimbingan akademik, konseling, atau bantuan keuangan jika diperlukan.")
            else:
                st.success("**Rekomendasi:** Siswa ini menunjukkan performa baik. Tetap pantau perkembangan akademiknya secara berkala.")

        else:
            st.markdown("""
            <div style='background:#f1f5f9; border-radius:16px; padding:48px 32px; text-align:center; color:#94a3b8;'>
                <div style='font-size:3rem;'>🎯</div>
                <div style='font-family:Syne,sans-serif; font-size:1.1rem; font-weight:700; color:#64748b; margin-top:12px;'>Isi form di kiri</div>
                <div style='font-size:0.85rem; margin-top:8px;'>dan klik tombol Prediksi untuk melihat hasil</div>
            </div>
            """, unsafe_allow_html=True)