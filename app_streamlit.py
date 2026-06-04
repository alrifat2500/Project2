import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="HydroCheck — Prediksi Hidrasi",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# CUSTOM CSS — MODERN CLEAN THEME
# ===============================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

    /* ===== GLOBAL ===== */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background: #f0f7ff;
    }

    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a2540 0%, #0d3b6e 60%, #1a5276 100%);
        border-right: none;
    }

    [data-testid="stSidebar"] * {
        color: #e8f4fd !important;
    }

    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #7dd3fc !important;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-top: 1.2rem;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(125, 211, 252, 0.2) !important;
        margin: 0.8rem 0;
    }

    /* ===== HIDE DEFAULT HEADER ===== */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    
    /* ===== METRIC CARDS ===== */
    [data-testid="stMetric"] {
        background: white;
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 2px 12px rgba(10, 37, 64, 0.07);
        border: 1px solid rgba(10, 37, 64, 0.05);
    }

    /* ===== BUTTONS ===== */
    .stButton > button {
        background: linear-gradient(135deg, #0a2540 0%, #1565c0 100%);
        color: white;
        border: none;
        border-radius: 14px;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
        font-size: 1rem;
        padding: 0.75rem 2rem;
        letter-spacing: 0.03em;
        transition: all 0.25s;
        box-shadow: 0 4px 15px rgba(10, 37, 64, 0.25);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(10, 37, 64, 0.35);
    }

    /* ===== SELECTBOX ===== */
    [data-testid="stSelectbox"] > div > div {
        border-radius: 12px !important;
        border-color: #cbd5e1 !important;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* ===== SUCCESS / ERROR / INFO ===== */
    [data-testid="stAlert"] {
        border-radius: 16px;
        border: none;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 500;
    }

    /* ===== DIVIDER ===== */
    hr {
        border-color: #e2e8f0;
        margin: 1.5rem 0;
    }

    /* ===== CUSTOM COMPONENTS ===== */
    .hero-card {
        background: linear-gradient(135deg, #0a2540 0%, #1565c0 50%, #1976d2 100%);
        border-radius: 24px;
        padding: 2.5rem 2rem;
        color: white;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }

    .hero-card::before {
        content: "💧";
        position: absolute;
        right: 2rem;
        top: 50%;
        transform: translateY(-50%);
        font-size: 5rem;
        opacity: 0.15;
    }

    .hero-card h1 {
        font-size: 1.9rem;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
        line-height: 1.2;
    }

    .hero-card p {
        font-size: 0.95rem;
        opacity: 0.82;
        margin: 0;
        font-weight: 400;
    }

    .info-card {
        background: white;
        border-radius: 18px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 12px rgba(10, 37, 64, 0.06);
        border: 1px solid rgba(10, 37, 64, 0.05);
        border-left: 4px solid #2196f3;
    }

    .info-card h3 {
        font-size: 1rem;
        font-weight: 700;
        color: #0a2540;
        margin: 0 0 0.5rem 0;
    }

    .info-card p, .info-card li {
        font-size: 0.9rem;
        color: #475569;
        line-height: 1.7;
        margin: 0;
    }

    .result-good {
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        border: 2px solid #10b981;
    }

    .result-poor {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        border: 2px solid #ef4444;
    }

    .result-title {
        font-size: 1.6rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }

    .result-sub {
        font-size: 0.92rem;
        opacity: 0.75;
        margin: 0;
    }

    .tip-box {
        background: #f0f9ff;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-top: 0.8rem;
        border: 1px solid #bae6fd;
    }

    .tip-box p {
        font-size: 0.88rem;
        color: #0369a1;
        margin: 0;
        font-weight: 500;
        line-height: 1.6;
    }

    .stat-pill {
        display: inline-block;
        background: linear-gradient(135deg, #0a2540, #1565c0);
        color: white;
        border-radius: 100px;
        padding: 0.3rem 1rem;
        font-size: 0.8rem;
        font-weight: 700;
        font-family: 'Space Mono', monospace;
        margin: 0.2rem;
    }

    .section-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #94a3b8;
        margin-bottom: 0.4rem;
    }

    .input-panel {
        background: white;
        border-radius: 20px;
        padding: 1.8rem;
        box-shadow: 0 2px 16px rgba(10, 37, 64, 0.07);
        border: 1px solid rgba(10, 37, 64, 0.05);
        height: 100%;
    }

    .panel-title {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #1565c0;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .dev-card {
        background: linear-gradient(135deg, #0a2540, #1565c0);
        border-radius: 20px;
        padding: 2rem;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }

    .dev-card h2 {
        font-size: 1.4rem;
        font-weight: 800;
        margin: 1rem 0 0.3rem 0;
        color: white;
    }

    .dev-card p {
        font-size: 0.88rem;
        opacity: 0.75;
        margin: 0;
    }

    .badge {
        background: rgba(255,255,255,0.15);
        border-radius: 100px;
        padding: 0.3rem 1rem;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.3rem;
        backdrop-filter: blur(4px);
    }

    .fact-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.8rem;
        margin-top: 1rem;
    }

    .fact-item {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #e2e8f0;
        text-align: center;
    }

    .fact-num {
        font-family: 'Space Mono', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        color: #0a2540;
    }

    .fact-desc {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 0.2rem;
    }

    .water-level-bar {
        background: #e2e8f0;
        border-radius: 100px;
        height: 10px;
        margin: 0.5rem 0;
        overflow: hidden;
    }

    .water-level-fill {
        height: 100%;
        border-radius: 100px;
        background: linear-gradient(90deg, #2196f3, #0a2540);
    }

    .stPlotlyChart, [data-testid="stPlotlyChart"] {
        border-radius: 16px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ===============================
# LOAD DATA & MODEL
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("Daily_Water_Intake.csv")

@st.cache_resource
def load_model():
    return joblib.load("decision_tree_model.joblib")

df = load_data()
model = load_model()


# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    col1, col2, col3 = st.columns([1,5,1])
    with col2:
        try:
            st.image("img/Logo_SMK_Negeri_1_Purbalingga.png", width=180)
        except:
            st.markdown("💧")

    st.markdown("""
    <div style="text-align:center; padding-bottom: 1rem;">
        <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:1.3rem; font-weight:800; color:#7dd3fc; letter-spacing:0.02em;">HydroCheck</div>
        <div style="font-size:0.75rem; color:#94a3b8; font-weight:500; letter-spacing:0.08em; text-transform:uppercase; margin-top:2px;">Hydration Predictor</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(125,211,252,0.2); margin:0.5rem 0 1rem 0;'>", unsafe_allow_html=True)

    st.markdown("## ℹ️ Tentang Aplikasi")
    st.markdown("""
    <div style="font-size:0.85rem; color:#cbd5e1; line-height:1.7;">
    Aplikasi ini menggunakan model <b style="color:#7dd3fc;">Machine Learning</b> untuk memprediksi tingkat hidrasi tubuh berdasarkan data personal dan kondisi lingkungan.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(125,211,252,0.2); margin:1rem 0;'>", unsafe_allow_html=True)

    st.markdown("## 🤖 Model")
    st.markdown("""
    <div style="font-size:0.83rem; color:#cbd5e1; line-height:1.8;">
    ✅ &nbsp;<b style="color:#86efac;">Decision Tree</b><br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(125,211,252,0.2); margin:1rem 0;'>", unsafe_allow_html=True)

    st.markdown("## 📊 Dataset")
    st.markdown(f"""
    <div style="font-size:0.83rem; color:#cbd5e1; line-height:1.8;">
    📁 &nbsp;Daily Water Intake<br>
    📋 &nbsp;<b style="color:#fbbf24;">{len(df):,}</b> baris data<br>
    🔖 &nbsp;<b style="color:#fbbf24;">2</b> kelas (Good / Poor)<br>
    📌 &nbsp;<b style="color:#fbbf24;">6</b> fitur input
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(125,211,252,0.2); margin:1rem 0;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; font-size:0.72rem; color:#64748b; padding: 0.5rem 0;">
        © 2026 HydroCheck · SMK ML Project
    </div>
    """, unsafe_allow_html=True)


# ===============================
# TABS
# ===============================
tab1, tab2, tab3, tab4 = st.tabs([
    "  💧  Prediksi  ",
    "  📚  Informasi  ",
    "  📓  Notebook  ",
    "  👨‍💻  Developer  "
])


# ===============================
# TAB 1 — PREDIKSI
# ===============================
with tab1:

    # Hero Banner
    st.markdown("""
    <div class="hero-card">
        <h1>💧 Cek Tingkat Hidrasimu</h1>
        <p>Isi data di bawah ini lalu tekan tombol Prediksi untuk mengetahui apakah tubuhmu terhidrasi dengan baik.</p>
    </div>
    """, unsafe_allow_html=True)

    # Input Panel
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="panel-title">👤 Data Personal</div>
        """, unsafe_allow_html=True)

        umur = st.slider("🎂 Umur (tahun)", 10, 80, 25)

        berat = st.slider("⚖️ Berat Badan (kg)", 30.0, 120.0, 60.0, step=0.5)
                       
        gender = st.selectbox("🚻 Jenis Kelamin",["Laki-Laki", "Perempuan"])

    with col2:
        st.markdown("""
        <div class="panel-title">🌿 Gaya Hidup & Lingkungan</div>
        """, unsafe_allow_html=True)

        air = st.slider("🥤 Konsumsi Air per Hari (liter)", 0.5, 5.0, 2.0, step=0.1)

        aktivitas = st.selectbox("🏃 Tingkat Aktivitas Fisik", ["Rendah", "Sedang", "Tinggi"])

        cuaca = st.selectbox("🌤️ Kondisi Cuaca", ["Normal", "Panas", "Dingin"])

    st.markdown("<br>", unsafe_allow_html=True)

    # Tombol Prediksi
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        prediksi_btn = st.button("🔎 Prediksi Sekarang", use_container_width=True)

    if prediksi_btn:

        # Mapping
        gender_map   = {"Laki-Laki": "Male",    "Perempuan": "Female"}
        aktivitas_map = {"Rendah": "Low",        "Sedang": "Moderate",  "Tinggi": "High"}
        cuaca_map    = {"Panas": "Hot",           "Dingin": "Cold",      "Normal": "Normal"}

        data_input = pd.DataFrame([[
            umur,
            gender_map[gender],
            berat,
            air,
            aktivitas_map[aktivitas],
            cuaca_map[cuaca]
        ]], columns=["Age", "Gender", "Weight (kg)", "Daily Water Intake (liters)",
                     "Physical Activity Level", "Weather"])

        hasil = model.predict(data_input)[0]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 🎯 Hasil Prediksi")

        col_res1, col_res2 = st.columns([1.2, 1])

        with col_res1:
            if hasil == "Good":
                st.markdown("""
                <div class="result-good">
                    <div style="font-size:3rem;">✅</div>
                    <div class="result-title" style="color:#065f46;">Hidrasi Baik!</div>
                    <div class="result-sub" style="color:#047857;">Tubuh Anda terhidrasi dengan baik.</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                <div class="tip-box">
                    <p>💡 <b>Tips:</b> Pertahankan kebiasaan minum yang baik! Tetap minum secara rutin meskipun tidak merasa haus, terutama saat berolahraga atau cuaca panas.</p>
                </div>
                """, unsafe_allow_html=True)

            elif hasil == "Poor":
                st.markdown("""
                <div class="result-poor">
                    <div style="font-size:3rem;">⚠️</div>
                    <div class="result-title" style="color:#991b1b;">Hidrasi Buruk!</div>
                    <div class="result-sub" style="color:#b91c1c;">Tubuh Anda kekurangan cairan.</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                <div class="tip-box" style="background:#fff5f5; border-color:#fca5a5;">
                    <p style="color:#b91c1c;">💡 <b>Saran:</b> Segera tambah konsumsi air putih minimal 2–3 liter per hari. Hindari minuman berkafein dan manis yang memperparah dehidrasi.</p>
                </div>
                """, unsafe_allow_html=True)

        with col_res2:
            # Ringkasan input
            st.markdown("**📋 Ringkasan Input Anda:**")
            summary_data = {
                "Parameter": ["Umur", "Berat Badan", "Jenis Kelamin", "Konsumsi Air", "Aktivitas", "Cuaca"],
                "Nilai": [f"{umur} tahun", f"{berat} kg", gender, f"{air} liter/hari", aktivitas, cuaca]
            }
            st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

        # Indikator visual konsumsi air
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 💧 Indikator Konsumsi Air Harian")

        kebutuhan = 3.7 if gender == "Laki-Laki" else 2.7
        persen = min(air / kebutuhan * 100, 100)

        selisih = round(kebutuhan - air, 1)
        kekurangan = max(selisih, 0)
        delta_sign = "-" if selisih > 0 else "+"
        delta_color_style = "#ef4444" if selisih > 0 else "#10b981"

        st.markdown(f"""
        <div style="background:white; border-radius:18px; padding:1.4rem 1.8rem; box-shadow:0 2px 12px rgba(10,37,64,0.07); border:1px solid rgba(10,37,64,0.05); display:flex; justify-content:space-around; align-items:center; flex-wrap:wrap; gap:1rem;">
            <div style="text-align:center; flex:1; min-width:120px;">
                <div style="font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color:#94a3b8; margin-bottom:0.3rem;">💧 Konsumsi Anda</div>
                <div style="font-size:1.6rem; font-weight:800; color:#0a2540; font-family:'Space Mono',monospace;">{air} liter</div>
            </div>
            <div style="width:1px; background:#e2e8f0; height:3rem;"></div>
            <div style="text-align:center; flex:1; min-width:120px;">
                <div style="font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color:#94a3b8; margin-bottom:0.3rem;">🎯 Kebutuhan Harian</div>
                <div style="font-size:1.6rem; font-weight:800; color:#0a2540; font-family:'Space Mono',monospace;">{kebutuhan} liter</div>
            </div>
            <div style="width:1px; background:#e2e8f0; height:3rem;"></div>
            <div style="text-align:center; flex:1; min-width:120px;">
                <div style="font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color:#94a3b8; margin-bottom:0.3rem;">📉 Kekurangan</div>
                <div style="font-size:1.6rem; font-weight:800; color:#0a2540; font-family:'Space Mono',monospace;">{kekurangan} liter</div>
                <div style="font-size:0.82rem; font-weight:600; color:{delta_color_style}; margin-top:2px;">{delta_sign}{abs(selisih)} liter</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="water-level-bar">
            <div class="water-level-fill" style="width:{persen:.0f}%;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#64748b; margin-top:4px;">
            <span>0 liter</span>
            <span style="color:#1565c0; font-weight:700;">{persen:.0f}% dari kebutuhan harian</span>
            <span>{kebutuhan} liter</span>
        </div>
        """, unsafe_allow_html=True)

    # ===== CHART =====
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📈 Distribusi Konsumsi Air dalam Dataset")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        fig1, ax1 = plt.subplots(figsize=(6, 3.5))
        fig1.patch.set_facecolor('#f8fafc')
        ax1.set_facecolor('#f8fafc')

        good = df[df["Hydration Level"] == "Good"]["Daily Water Intake (liters)"]
        poor = df[df["Hydration Level"] == "Poor"]["Daily Water Intake (liters)"]

        ax1.hist(good, bins=30, alpha=0.75, color='#2196f3', label='Good', edgecolor='white')
        ax1.hist(poor, bins=30, alpha=0.75, color='#ef4444', label='Poor', edgecolor='white')
        ax1.set_xlabel("Konsumsi Air (liter)", fontsize=9, color='#475569')
        ax1.set_ylabel("Jumlah", fontsize=9, color='#475569')
        ax1.set_title("Distribusi Konsumsi Air per Kelas", fontsize=10, fontweight='bold', color='#0a2540', pad=12)
        ax1.legend(fontsize=8)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.tick_params(labelsize=8, colors='#64748b')
        for spine in ['left', 'bottom']:
            ax1.spines[spine].set_color('#e2e8f0')
        plt.tight_layout()
        st.pyplot(fig1)

    with col_chart2:
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        fig2.patch.set_facecolor('#f8fafc')
        ax2.set_facecolor('#f8fafc')

        counts = df["Hydration Level"].value_counts()
        colors = ['#2196f3', '#ef4444']
        wedges, texts, autotexts = ax2.pie(
            counts.values,
            labels=counts.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=2)
        )
        for text in texts:
            text.set_fontsize(9)
            text.set_color('#475569')
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')
            autotext.set_color('white')
        ax2.set_title("Proporsi Kelas Hidrasi", fontsize=10, fontweight='bold', color='#0a2540', pad=12)
        plt.tight_layout()
        st.pyplot(fig2)


# ===============================
# TAB 2 — INFORMASI
# ===============================
with tab2:

    st.markdown("""
    <div class="hero-card">
        <h1>📚 Panduan Lengkap Hidrasi</h1>
        <p>Pelajari semua yang perlu kamu tahu tentang hidrasi tubuh dan dampaknya pada kesehatan.</p>
    </div>
    """, unsafe_allow_html=True)

    # Fakta cepat
    st.markdown("### ⚡ Fakta Singkat Hidrasi")
    st.markdown("""
    <div class="fact-grid">
        <div class="fact-item">
            <div class="fact-num">60%</div>
            <div class="fact-desc">Tubuh manusia terdiri dari air</div>
        </div>
        <div class="fact-item">
            <div class="fact-num">2.7L</div>
            <div class="fact-desc">Kebutuhan air harian wanita dewasa</div>
        </div>
        <div class="fact-item">
            <div class="fact-num">3.7L</div>
            <div class="fact-desc">Kebutuhan air harian pria dewasa</div>
        </div>
        <div class="fact-item">
            <div class="fact-num">75%</div>
            <div class="fact-desc">Otak manusia tersusun dari air</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Apa itu hidrasi
    st.markdown("### 💧 Apa itu Hidrasi?")
    st.markdown("""
    <div class="info-card">
        <h3>🔬 Definisi</h3>
        <p>Hidrasi adalah kondisi keseimbangan cairan di dalam tubuh. Tubuh membutuhkan air untuk <b>hampir semua fungsi biologis</b>, mulai dari mengangkut nutrisi, mengatur suhu, melancarkan pencernaan, hingga menjaga konsentrasi dan mood. Kekurangan cairan bahkan 1–2% saja sudah bisa mengganggu fungsi kognitif dan fisik.</p>
        <hr style="border-color:#e2e8f0; margin:1rem 0;">
        <div style="display:flex; gap:1.5rem; flex-wrap:wrap;">
            <div style="flex:1; min-width:200px;">
                <div style="font-weight:700; color:#ef4444; margin-bottom:0.5rem;">🚨 Tanda-Tanda Dehidrasi</div>
                <p style="margin:0;">
                ❌ &nbsp;Mulut dan tenggorokan kering<br>
                ❌ &nbsp;Urin berwarna kuning gelap<br>
                ❌ &nbsp;Sakit kepala dan pusing<br>
                ❌ &nbsp;Mudah lelah dan lemas<br>
                ❌ &nbsp;Sulit berkonsentrasi<br>
                ❌ &nbsp;Jarang buang air kecil<br>
                ❌ &nbsp;Kulit kering dan tidak elastis
                </p>
            </div>
            <div style="width:1px; background:#e2e8f0;"></div>
            <div style="flex:1; min-width:200px;">
                <div style="font-weight:700; color:#10b981; margin-bottom:0.5rem;">✅ Tanda-Tanda Hidrasi Baik</div>
                <p style="margin:0;">
                ✔️ &nbsp;Urin berwarna kuning muda / jernih<br>
                ✔️ &nbsp;Kulit lembap dan elastis<br>
                ✔️ &nbsp;Energi stabil sepanjang hari<br>
                ✔️ &nbsp;Konsentrasi dan fokus baik<br>
                ✔️ &nbsp;Buang air kecil 6–8x per hari<br>
                ✔️ &nbsp;Tidak sering pusing<br>
                ✔️ &nbsp;Nafas segar
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Faktor yang mempengaruhi
    st.markdown("### 📌 Faktor yang Mempengaruhi Kebutuhan Air")
    st.markdown("""
    <div class="info-card">
        <div style="display:flex; flex-wrap:wrap; gap:1.2rem;">
            <div style="flex:1; min-width:160px; border-left:3px solid #f59e0b; padding-left:0.8rem;">
                <div style="font-weight:700; color:#0a2540; margin-bottom:0.4rem;">👤 Usia & Jenis Kelamin</div>
                <p style="margin:0; font-size:0.88rem;">
                • Pria butuh <b>~3.7 liter/hari</b><br>
                • Wanita butuh <b>~2.7 liter/hari</b><br>
                • Lansia lebih rentan dehidrasi<br>
                • Anak-anak butuh proporsional terhadap berat badan
                </p>
            </div>
            <div style="flex:1; min-width:160px; border-left:3px solid #8b5cf6; padding-left:0.8rem;">
                <div style="font-weight:700; color:#0a2540; margin-bottom:0.4rem;">🏃 Aktivitas Fisik</div>
                <p style="margin:0; font-size:0.88rem;">
                • <b>Rendah</b>: duduk, belajar → kebutuhan standar<br>
                • <b>Sedang</b>: jalan kaki, kerja → +0.5L<br>
                • <b>Tinggi</b>: olahraga berat → +1–2L<br>
                • Minum 200ml setiap 20 menit saat olahraga
                </p>
            </div>
            <div style="flex:1; min-width:160px; border-left:3px solid #ec4899; padding-left:0.8rem;">
                <div style="font-weight:700; color:#0a2540; margin-bottom:0.4rem;">🌤️ Cuaca & Suhu</div>
                <p style="margin:0; font-size:0.88rem;">
                • <b>Panas</b> (&gt;30°C): tambah 1–1.5L<br>
                • <b>Normal</b> (20–30°C): kebutuhan standar<br>
                • <b>Dingin</b> (&lt;20°C): tetap minum meski tidak haus<br>
                • Kelembaban tinggi meningkatkan keringat
                </p>
            </div>
            <div style="flex:1; min-width:160px; border-left:3px solid #14b8a6; padding-left:0.8rem;">
                <div style="font-weight:700; color:#0a2540; margin-bottom:0.4rem;">⚖️ Berat Badan</div>
                <p style="margin:0; font-size:0.88rem;">
                Rumus: <b>berat badan × 0.033 liter/kg</b><br><br>
                • 50 kg → 1.65 liter<br>
                • 70 kg → 2.31 liter<br>
                • 90 kg → 2.97 liter
                </p>
            </div>
            <div style="flex:1; min-width:160px; border-left:3px solid #f97316; padding-left:0.8rem;">
                <div style="font-weight:700; color:#0a2540; margin-bottom:0.4rem;">🍎 Makanan & Minuman</div>
                <p style="margin:0; font-size:0.88rem;">
                • ~20% kebutuhan air bisa dari makanan<br>
                • Kafein dan alkohol bersifat diuretik<br>
                • Buah tinggi air: semangka, mentimun, jeruk<br>
                • Hindari minuman manis berlebihan
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tips praktis
    st.markdown("### 💡 Tips Menjaga Hidrasi Setiap Hari")
    st.markdown("""
    <div class="info-card" style="border-left-color: #2196f3; background: linear-gradient(135deg, #f0f9ff, #e0f2fe);">
        <h3>📅 Jadwal Minum yang Dianjurkan</h3>
        <p>
        ☀️ &nbsp;<b>Bangun tidur</b> → minum 1–2 gelas segera setelah bangun<br>
        🍳 &nbsp;<b>Sebelum makan</b> → minum 1 gelas untuk membantu pencernaan<br>
        🌞 &nbsp;<b>Pagi–siang</b> → minum setiap 1 jam sekali<br>
        🏃 &nbsp;<b>Saat olahraga</b> → minum 200ml setiap 20 menit<br>
        🌙 &nbsp;<b>Sebelum tidur</b> → minum 1 gelas air putih<br><br>
        💡 Gunakan botol minum berukuran 600ml–1L dan targetkan habis beberapa kali sehari.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Visualisasi aktivitas vs kebutuhan air
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Visualisasi Kebutuhan Air Berdasarkan Aktivitas & Cuaca")

    fig3, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig3.patch.set_facecolor('#f8fafc')

    # Chart aktivitas
    aktivitas_labels = ['Rendah', 'Sedang', 'Tinggi']
    kebutuhan_pria   = [3.7, 4.2, 5.0]
    kebutuhan_wanita = [2.7, 3.2, 4.0]
    x = np.arange(len(aktivitas_labels))
    width = 0.35

    axes[0].set_facecolor('#f8fafc')
    bars1 = axes[0].bar(x - width/2, kebutuhan_pria,   width, label='Laki-Laki', color='#2196f3', alpha=0.85, edgecolor='white', linewidth=1.5)
    bars2 = axes[0].bar(x + width/2, kebutuhan_wanita, width, label='Perempuan', color='#ec4899', alpha=0.85, edgecolor='white', linewidth=1.5)
    axes[0].set_xlabel('Tingkat Aktivitas', fontsize=9, color='#475569')
    axes[0].set_ylabel('Kebutuhan Air (liter)', fontsize=9, color='#475569')
    axes[0].set_title('Kebutuhan Air per Aktivitas & Gender', fontsize=10, fontweight='bold', color='#0a2540', pad=10)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(aktivitas_labels, fontsize=8)
    axes[0].legend(fontsize=8)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].tick_params(labelsize=8, colors='#64748b')
    for spine in ['left', 'bottom']:
        axes[0].spines[spine].set_color('#e2e8f0')
    for bar in list(bars1) + list(bars2):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f'{bar.get_height()}L', ha='center', fontsize=7.5, color='#475569', fontweight='bold')

    # Chart cuaca
    cuaca_labels   = ['Dingin', 'Normal', 'Panas']
    tambahan_air   = [0, 0.5, 1.5]
    colors_cuaca   = ['#7dd3fc', '#2196f3', '#f97316']
    axes[1].set_facecolor('#f8fafc')
    bars3 = axes[1].bar(cuaca_labels, tambahan_air, color=colors_cuaca, alpha=0.85, edgecolor='white', linewidth=1.5, width=0.5)
    axes[1].set_xlabel('Kondisi Cuaca', fontsize=9, color='#475569')
    axes[1].set_ylabel('Tambahan Kebutuhan Air (liter)', fontsize=9, color='#475569')
    axes[1].set_title('Tambahan Kebutuhan Air per Cuaca', fontsize=10, fontweight='bold', color='#0a2540', pad=10)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].tick_params(labelsize=8, colors='#64748b')
    for spine in ['left', 'bottom']:
        axes[1].spines[spine].set_color('#e2e8f0')
    for bar, val in zip(bars3, tambahan_air):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'+{val}L', ha='center', fontsize=9, color='#475569', fontweight='bold')

    plt.tight_layout(pad=2)
    st.pyplot(fig3)

    # Dampak dehidrasi
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ⚠️ Dampak Dehidrasi pada Tubuh")

    st.markdown("""
    <div class="info-card">
        <div style="display:flex; flex-wrap:wrap; gap:1.2rem;">
            <div style="flex:1; min-width:180px; border-left:3px solid #f59e0b; padding-left:0.8rem;">
                <div style="font-weight:700; color:#0a2540; margin-bottom:0.4rem;">🧠 Otak & Mental</div>
                <p style="margin:0; font-size:0.88rem;">
                Kehilangan 1–2% cairan menurunkan konsentrasi, memori jangka pendek, dan meningkatkan risiko sakit kepala serta kelelahan mental.
                </p>
            </div>
            <div style="width:1px; background:#e2e8f0;"></div>
            <div style="flex:1; min-width:180px; border-left:3px solid #ef4444; padding-left:0.8rem;">
                <div style="font-weight:700; color:#0a2540; margin-bottom:0.4rem;">💪 Otot & Fisik</div>
                <p style="margin:0; font-size:0.88rem;">
                Dehidrasi menyebabkan kram otot, penurunan kekuatan fisik, dan pemulihan lebih lambat setelah olahraga.
                </p>
            </div>
            <div style="width:1px; background:#e2e8f0;"></div>
            <div style="flex:1; min-width:180px; border-left:3px solid #8b5cf6; padding-left:0.8rem;">
                <div style="font-weight:700; color:#0a2540; margin-bottom:0.4rem;">🫀 Organ Vital</div>
                <p style="margin:0; font-size:0.88rem;">
                Jangka panjang: risiko batu ginjal, infeksi saluran kemih, dan gangguan jantung meningkat signifikan.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ===============================
# TAB 4 — DEVELOPER
# ===============================
with tab4:

    st.markdown("""
    <div class="hero-card">
        <h1>👨‍💻 Tentang Developer</h1>
        <p>Informasi tentang pembuat aplikasi dan teknologi yang digunakan.</p>
    </div>
    """, unsafe_allow_html=True)

    col_dev1, col_dev2 = st.columns([1, 1.5], gap="large")

    with col_dev1:
        st.markdown("""
        <div class="dev-card">
            <div style="font-size:4rem;">👨‍🎓</div>
            <h2>‎ ‎ ‎ ‎ ‎ Alrifat</h2>
            <p>Machine Learning Student</p>
            <br>
            <span class="badge">🎓 SMK</span>
            <span class="badge">🤖 Machine Learning</span>
            <span class="badge">🐍 Python</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card" style="border-left-color:#10b981; margin-top: 0.8rem;">
            <h3>📬 Kontak</h3>
            <p>
            🐙 &nbsp;<a href="https://github.com/alrifat2500" target="_blank" style="color:#2196f3; text-decoration:none; font-weight:600;">github.com/alrifat2500</a><br><br>
            <a href="https://wa.me/6282136829324" target="_blank" style="display:inline-flex; align-items:center; gap:0.5rem; background:#25D366; color:white; padding:0.45rem 1.1rem; border-radius:100px; font-size:0.85rem; font-weight:700; text-decoration:none; margin-top:0.3rem;">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                  <path d="M13.601 2.326A7.85 7.85 0 0 0 7.994 0C3.627 0 .068 3.558.064 7.926c0 1.399.366 2.76 1.057 3.965L0 16l4.204-1.102a7.9 7.9 0 0 0 3.79.965h.004c4.368 0 7.926-3.558 7.93-7.93A7.9 7.9 0 0 0 13.6 2.326zM7.994 14.521a6.6 6.6 0 0 1-3.356-.92l-.24-.144-2.494.654.666-2.433-.156-.251a6.56 6.56 0 0 1-1.007-3.505c0-3.626 2.957-6.584 6.591-6.584a6.56 6.56 0 0 1 4.66 1.931 6.56 6.56 0 0 1 1.928 4.66c-.004 3.639-2.961 6.592-6.592 6.592m3.615-4.934c-.197-.099-1.17-.578-1.353-.646-.182-.065-.315-.099-.445.099-.133.197-.513.646-.627.775-.114.133-.232.148-.43.05-.197-.1-.836-.308-1.592-.985-.59-.525-.985-1.175-1.103-1.372-.114-.198-.011-.304.088-.403.087-.088.197-.232.296-.346.1-.114.133-.198.198-.33.065-.134.034-.248-.015-.347-.05-.099-.445-1.076-.612-1.47-.16-.389-.323-.335-.445-.34-.114-.007-.247-.007-.38-.007a.73.73 0 0 0-.529.247c-.182.198-.691.677-.691 1.654s.71 1.916.81 2.049c.098.133 1.394 2.132 3.383 2.992.47.205.84.326 1.129.418.475.152.904.129 1.246.08.38-.058 1.171-.48 1.338-.943.164-.464.164-.86.114-.943-.049-.084-.182-.133-.38-.232"/>
                </svg>
                Chat WhatsApp
            </a>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_dev2:
        st.markdown("### 🛠️ Teknologi yang Digunakan")

        tech_items = [
            ("🐍", "Python 3.x",         "Bahasa pemrograman utama"),
            ("📊", "Pandas",              "Manipulasi dan analisis data"),
            ("🤖", "Scikit-learn",        "Algoritma Machine Learning"),
            ("💾", "Joblib",              "Menyimpan dan memuat model"),
            ("📈", "Matplotlib / Seaborn","Visualisasi data"),
            ("🌐", "Streamlit",           "Framework web app"),
        ]

        for icon, name, desc in tech_items:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:1rem; background:white; border-radius:12px; padding:0.8rem 1rem; margin-bottom:0.5rem; box-shadow:0 1px 6px rgba(10,37,64,0.06); border:1px solid rgba(10,37,64,0.05);">
                <div style="font-size:1.5rem; min-width:2rem; text-align:center;">{icon}</div>
                <div>
                    <div style="font-weight:700; font-size:0.88rem; color:#0a2540;">{name}</div>
                    <div style="font-size:0.78rem; color:#64748b;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🤖 Model Machine Learning")

        models_info = [
            ("🔵", "Logistic Regression", "99.67%", "Regresi linear untuk klasifikasi biner"),
            ("🌲", "Random Forest",       "98.92%", "Ensemble dari banyak decision tree"),
            ("🌳", "Decision Tree",       "99.75%", "Pohon keputusan berbasis aturan if-else"),
        ]

        for icon, name, acc, desc in models_info:
            st.markdown(f"""
            <div style="display:flex; align-items:center; justify-content:space-between; background:white; border-radius:12px; padding:0.8rem 1rem; margin-bottom:0.5rem; box-shadow:0 1px 6px rgba(10,37,64,0.06); border:1px solid rgba(10,37,64,0.05);">
                <div style="display:flex; align-items:center; gap:0.8rem;">
                    <span style="font-size:1.3rem;">{icon}</span>
                    <div>
                        <div style="font-weight:700; font-size:0.88rem; color:#0a2540;">{name}</div>
                        <div style="font-size:0.75rem; color:#64748b;">{desc}</div>
                    </div>
                </div>
                <span style="background:linear-gradient(135deg,#0a2540,#1565c0); color:white; border-radius:100px; padding:0.2rem 0.7rem; font-size:0.75rem; font-weight:700; font-family:'Space Mono',monospace;">{acc}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    <div style="text-align:center; padding:1.5rem; background:white; border-radius:16px; box-shadow:0 2px 12px rgba(10,37,64,0.06); border:1px solid rgba(10,37,64,0.05);">
        <div style="font-size:0.8rem; color:#94a3b8; font-weight:500;">
            💧 HydroCheck &nbsp;·&nbsp; SMK Machine Learning Project &nbsp;·&nbsp; 2026<br>
            <span style="font-size:0.72rem; font-family:'Space Mono',monospace; margin-top:4px; display:block;">Built with Python · Scikit-learn · Streamlit</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# TAB 3 — NOTEBOOK
# ===============================
with tab3:
    st.markdown("""
    <div class="hero-card">
        <h1>📓 Notebook Machine Learning</h1>
        <p>Menampilkan kode dari Jupyter Notebook beserta outputnya.</p>
    </div>
    """, unsafe_allow_html=True)

    import nbformat

    try:
        nb = nbformat.read("klasifikasi.ipynb", as_version=4)

        for i, cell in enumerate(nb.cells):
            st.markdown(f"### Cell {i+1} ({cell.cell_type})")

            if cell.cell_type == "code":
                st.code(cell.source, language="python")

                if cell.get("outputs"):
                    st.markdown("**Output:**")

                    for output in cell["outputs"]:
                        if output.output_type == "stream":
                            st.text(output.text)

                        elif output.output_type in ["execute_result", "display_data"]:
                            data = output.get("data", {})

                            if "text/plain" in data:
                                st.text(data["text/plain"])

                            elif "text/html" in data:
                                st.markdown(data["text/html"], unsafe_allow_html=True)

                        elif output.output_type == "error":
                            st.error("\n".join(output.get("traceback", [])))

            elif cell.cell_type == "markdown":
                st.markdown(cell.source)

            st.divider()

    except Exception as e:
        st.error(f"Gagal memuat notebook: {e}")
