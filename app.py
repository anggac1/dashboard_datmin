import streamlit as st
import pandas as pd
import joblib
import os 
import altair as alt # Library grafik bawaan Streamlit
import re # Untuk preprocessing sederhana
from deep_translator import GoogleTranslator

# ==========================================
# ⚙️ 1. KONFIGURASI HALAMAN & PATH
# ==========================================
st.set_page_config(
    page_title="Final Project Datmin - Email Spam Analysis", 
    layout="wide", 
    page_icon="📊"
)

# --- KONFIGURASI PATH ---
# Mode Lokal (Laptop)
# BASE_PATH = r"C:\Users\DELL\Downloads\ads" 
# Mode Deploy (GitHub) -> Uncomment baris bawah saat upload nanti
BASE_PATH = "." 

# ==========================================
# 📥 2. FUNGSI LOAD DATA, MODEL & PREPROCESSING
# ==========================================

@st.cache_data
def load_rekap_data():
    """Memuat data rekap CSV"""
    file_path = os.path.join(BASE_PATH, 'data', 'rekap_performa.csv')
    try:
        df = pd.read_csv(file_path)
        # Bersihkan nama kolom dari spasi
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        return pd.DataFrame()

def load_model_dynamic(model_name, nlp_name, split_name):
    map_model = {"Random Forest": "rf", "Logistic Regression": "lr"}
    map_nlp   = {"TF-IDF": "tfidf", "BoW": "bow"}
    map_split = {"K-Fold CV": "kfcv", "Repeated Holdout": "rh"} 

    kode_m = map_model.get(model_name, "rf")
    kode_n = map_nlp.get(nlp_name, "tfidf")
    kode_s = map_split.get(split_name, "rh")

    filename = f"{kode_m}_{kode_n}_{kode_s}.pkl"
    model_path = os.path.join(BASE_PATH, 'models', filename)
    
    # Generate juga nama file gambar untuk logic nanti
    img_cm = f"cm_{kode_m}_{kode_n}_{kode_s}.png"
    img_roc = f"roc_{kode_m}_{kode_n}_{kode_s}.png"
    
    try:
        loaded_model = joblib.load(model_path)
        return loaded_model, filename, img_cm, img_roc
    except FileNotFoundError:
        return None, filename, None, None

def simple_preprocessing(text):
    """
    Preprocessing ringan agar input user cocok dengan vocabulary model.
    1. Lowercase
    2. Hapus karakter selain huruf dan angka
    """
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text) 
    return text

# 🗂️ 3. SIDEBAR NAVIGASI
st.sidebar.title("🗂️ Spam Detector App")
menu = st.sidebar.radio("Menu Utama:",
    ["1. Introduction", "2. Visualisasi Data", "3. Evaluasi Data", "4. Model Benchmarking"]
)
st.sidebar.markdown("---")
st.sidebar.caption(
    """
    Made with 💙 from Group 13 Datmin K Gasal 2025/2026
    
    **Anggota Kelompok:**
    1. Ahmad Al-Rifai (5003221108)
    2. M. Irfan Kurniawan (5003231194)
    3. Aditya Wahib Erlangga (5003231206)
    """
)
# 📖 4. KONTEN HALAMAN
# HALAMAN 1: INTRODUCTION
if menu == "1. Introduction":
    st.title("📑 Final Project: E-mail Spam Classification Analysis")
    st.subheader("Mata Kuliah Data Mining dan Visualisasi")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Latar Belakang")
        st.write("""
        Email merupakan salah satu sarana komunikasi utama, namun rentan terhadap penyalahgunaan berupa *spam*. 
        Spam tidak hanya mengganggu, tetapi juga dapat membawa risiko keamanan seperti *phishing* dan penyebaran malware.
        Sebagai upaya mitigasi, klasifikasi email menjadi spam atau ham (aman) sangat penting dilakukan.
                 
        Dalam Project kali ini, akan dicari 8 kombinasi model dalam mengklasifikasikan email spam dan ham (bukan spam).
        Selain itu, akan ada juga fitur prediksi langsung berdasarkan input teks email dari pengguna.
        """)
        
        st.markdown("### Tentang Dataset")
        st.info("""
        **Sumber Data:** [Kaggle - Email Spam Classification Dataset](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv)
        
        Dataset ini berisi **5.172** baris data email dengan 3002 kolom, dimana:
        * **Email No.**: Nomor urut email pada kolom pertama.
        * **Prediction**: Kolom Terakhir, Kategori pesan Email (1 untuk Spam, 0 untuk Ham/Aman).
        * Sisa 3000 kolom lainnya berisi 3000 kata paling populer pada email.
        
        Dataset ini dipilih karena memiliki variasi teks yang cukup untuk menguji ketahanan model NLP.
        """)

        st.markdown("### Tujuan Penelitian")
        st.write("""
        1. **Membandingkan Metode NLP:** Efektivitas representasi teks menggunakan *TF-IDF* vs *Bag of Words*.
        2. **Analisis Validasi:** Dampak penggunaan *K-Fold Cross Validation* dibandingkan *Repeated Holdout*.
        3. **Evaluasi Model:** Membandingkan performa *Random Forest* dan *Logistic Regression*.
        """)
        
    with col2:
        try:
            img_intro = os.path.join(BASE_PATH, 'assets', 'intro.png')
            st.image(img_intro, use_container_width=True)
        except:
            pass

# ------------------------------------------
# HALAMAN 2: VISUALISASI DATA
# ------------------------------------------
elif menu == "2. Visualisasi Data":
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.write("Tahap ini bertujuan untuk memahami karakteristik data sebelum dilakukan pemodelan.")
    
    tab1, tab2 = st.tabs(["Distribusi Label", "Wordcloud Analysis"])
    
    with tab1:
        st.subheader("Keseimbangan Data (Imbalance Check)")
        try:
            img_bar = os.path.join(BASE_PATH, 'assets', 'sebaran.png')
            st.image(img_bar, width=600)
            st.caption("""
            **Analisis:** Grafik di atas menunjukkan perbandingan jumlah email Spam dan Ham. 
            Terlihat bahwa data tidak seimbang, dengan email Ham (Aman) jauh lebih banyak dibandingkan Spam.
            """)
        except:
            st.error("File 'sebaran.png' tidak ditemukan di folder assets.")
            
    with tab2:
        st.subheader("Analisis Kata Kunci (Wordcloud)")
        st.write("Visualisasi kata-kata yang paling sering muncul (frekuensi tinggi) pada masing-masing kategori.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🟥 Wordcloud SPAM**")
            try:
                st.image(os.path.join(BASE_PATH, 'assets', 'wordcloudspam.png'), use_container_width=True)
            except: st.write("Gambar tidak tersedia.")
        with c2:
            st.markdown("**🟩 Wordcloud HAM**")
            try:
                st.image(os.path.join(BASE_PATH, 'assets', 'wordcloudham.png'), use_container_width=True)
            except: st.write("Gambar tidak tersedia.")

# HALAMAN 3: EVALUASI DATA
elif menu == "3. Evaluasi Data":
    st.title("📈 Evaluasi Performa Model Keseluruhan")
    st.write("Halaman ini menyajikan ringkasan performa dari seluruh kombinasi eksperimen.\n")
    df = load_rekap_data()

    if not df.empty:
        # --- 1. LEADERBOARD TABLE ---
        st.subheader("🏆 Leaderboard Model")
        st.caption("Secara default tabel diurutkan berdasarkan **F1-Score**. Anda dapat mengklik header kolom lain untuk mengurutkan ulang.")
        
        # Urutkan berdasarkan nama kolom ASLI di CSV
        leaderboard = df.sort_values(by='F1_Score', ascending=False)
        
        if 'Report' in leaderboard.columns:
            leaderboard_clean = leaderboard.drop(columns=['Report'])
        else:
            leaderboard_clean = leaderboard

        st.dataframe(
            leaderboard_clean, 
            use_container_width=True,
            hide_index=True,
            column_config={
                # KUNCI (Kiri) harus nama kolom ASLI di CSV
                # LABEL (Kanan) adalah nama cantik yang ingin ditampilkan
                "Test_Acc": st.column_config.ProgressColumn("Akurasi (Test)", format="%.3f", min_value=0, max_value=1),
                "Train_Acc": st.column_config.ProgressColumn("Akurasi (Train)", format="%.3f", min_value=0, max_value=1),
                "F1_Score": st.column_config.ProgressColumn("F1-Score", format="%.3f", min_value=0, max_value=1),
                "ROC_AUC": st.column_config.NumberColumn("ROC-AUC", format="%.3f"),
                "Precision": st.column_config.NumberColumn("Precision", format="%.3f"),
                "Recall": st.column_config.NumberColumn("Recall", format="%.3f")
            }
        )

        st.divider()

        # --- 2. GRAFIK ALTAIR ---
        st.subheader("📊 Perbandingan Metrik Antar Skenario")
        
        # Mapping: Nama Cantik (Pilihan User) -> Nama Kolom Asli (Dataframe)
        metric_map = {
            "F1-Score": "F1_Score",
            "Testing Accuracy": "Test_Acc",
            "Training Accuracy": "Train_Acc",
            "ROC-AUC": "ROC_AUC",
            "Precision": "Precision",
            "Recall": "Recall"
        }
        
        # Selectbox menampilkan Nama Cantik
        metric_label = st.selectbox("Pilih Metrik untuk Grafik:", list(metric_map.keys()))
        
        # Ambil Nama Kolom Asli untuk plotting
        metric_col = metric_map[metric_label]
        
        # Buat kolom Label gabungan untuk sumbu Y
        df['Label'] = df['Model'] + " + " + df['NLP'] + " (" + df['Split'] + ")"
        
        chart = alt.Chart(df).mark_bar().encode(
            # x menggunakan kolom asli (metric_col), tapi title menggunakan nama cantik (metric_label)
            x=alt.X(metric_col, title=f"Nilai {metric_label}"),
            y=alt.Y('Label', sort='-x', title="Skenario Eksperimen"),
            color=alt.Color('Model', legend=alt.Legend(title="Algoritma")),
            tooltip=['Label', 'Model', alt.Tooltip(metric_col, title=metric_label, format=".3f")]
        ).properties(
            height=500
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

    else:
        st.warning("Data CSV kosong atau tidak ditemukan.")# Tambahkan import ini di bagian paling atas app.py

# ... (Kode konfigurasi dan load data biarkan saja) ...

# ------------------------------------------
# HALAMAN 4: MODEL BENCHMARKING (DENGAN TRANSLATOR)
# ------------------------------------------
elif menu == "4. Model Benchmarking":
    st.title("⚙️ Detail Model & Prediksi")
    st.markdown(
        """
        Analisis mendalam per skenario, visualisasi evaluasi, dan uji coba prediksi.
        
        Silakan pilih kombinasi model yang tersedia di bawah ini:
        """
    )

    # A. SELEKSI MODEL
    col1, col2, col3 = st.columns(3)
    with col1:
        nlp_opt = st.selectbox("Fitur NLP", ["TF-IDF", "BoW"])
    with col2:
        split_opt = st.selectbox("Metode Split", ["K-Fold CV", "Repeated Holdout"])
    with col3:
        model_opt = st.selectbox("Algoritma", ["Random Forest", "Logistic Regression"])

    st.divider()

    # B. LOAD DATA & MODEL
    df = load_rekap_data()
    model, filename, img_cm_name, img_roc_name = load_model_dynamic(model_opt, nlp_opt, split_opt)

    # Filter Dataframe
    subset = df[
        (df['NLP'] == nlp_opt) & 
        (df['Split'] == split_opt) & 
        (df['Model'] == model_opt)
    ]

    # --- TABS LAYOUT ---
    tab_metrik, tab_grafik, tab_prediksi = st.tabs(["📊 Metrik & Laporan", "📈 Grafik Evaluasi", "🤖 Live Prediction"])

    # === TAB 1: METRIK & REPORT ===
    with tab_metrik:
        if not subset.empty:
            row = subset.iloc[0]
            st.subheader("Summary Metrics")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Training Accuracy", f"{row['Train_Acc']*100:.2f}%")
            c2.metric("Testing Accuracy",  f"{row['Test_Acc']*100:.2f}%")
            c3.metric("F1-Score",          f"{row['F1_Score']:.3f}")
            c4.metric("ROC-AUC",           f"{row['ROC_AUC']:.3f}")
            
            diff = row['Train_Acc'] - row['Test_Acc']
            if diff > 0.05:
                st.warning(f"⚠️ Potensi Overfitting: Akurasi Training lebih tinggi {diff*100:.1f}% daripada Testing.")
            else:
                st.success("✅ Model Stabil (Good Fit)")
            
            st.divider()
            st.subheader("Classification Report")
            st.code(row['Report'], language='text')
        else:
            st.warning("Data kombinasi ini tidak ditemukan di CSV.")

    # === TAB 2: GRAFIK (CM & ROC) ===
    with tab_grafik:
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.write("**Confusion Matrix**")
            try:
                st.image(os.path.join(BASE_PATH, 'assets', img_cm_name), use_container_width=True)
            except: st.write(f"Gambar {img_cm_name} tidak ditemukan.")
        with col_g2:
            st.write("**ROC-AUC Curve**")
            try:
                st.image(os.path.join(BASE_PATH, 'assets', img_roc_name), use_container_width=True)
            except: st.write(f"Gambar {img_roc_name} tidak ditemukan.")

    # === TAB 3: LIVE PREDICTION (MULTI-LANGUAGE) ===
    with tab_prediksi:
        if model:
            st.info(f"Menggunakan Model Aktif: `{filename}`")
            # REVISI KATA-KATA: Menjadi Multi-Bahasa
            st.markdown("""
            💡 **Fitur Cerdas:** Model ini mendukung **Multi-Bahasa** (Indonesia, Inggris, Spanyol, Daerah, dll).
            Sistem akan otomatis menerjemahkan input Anda ke Bahasa Inggris sebelum melakukan prediksi.
            """)
            
            text_input = st.text_area("Masukkan Subject/Body Email:", height=100, 
                                     placeholder="Contoh: SELAMAT! Anda memenangkan hadiah undian Rp 1 Milyar. Klik di sini sekarang!")
            
            if st.button("Analisis Email"):
                if text_input:
                    # PROSES 1: TRANSLASI (AUTO -> EN)
                    with st.spinner("Menerjemahkan & Menganalisis..."):
                        try:
                            # Auto-detect language and translate to English
                            translated_text = GoogleTranslator(source='auto', target='en').translate(text_input)
                            
                            # Tampilkan hasil terjemahan agar user tahu (Transparansi)
                            st.caption(f"📝 **Terjemahan Sistem:** *{translated_text}*")
                            
                        except Exception as e:
                            st.error("Gagal menerjemahkan (Cek koneksi internet). Menggunakan teks asli.")
                            translated_text = text_input

                    # PROSES 2: PREPROCESSING
                    clean_input = simple_preprocessing(translated_text)
                    
                    try:
                        # PROSES 3: PREDIKSI
                        prediksi = model.predict([clean_input])[0]
                        
                        # 4. TAMPILKAN HASIL DENGAN EFEK
                        if prediksi == 1 or prediksi == 'spam': 
                            st.snow() # Efek Salju
                            st.error("🚨 KESIMPULAN: SPAM DETECTED!")
                            st.toast("⚠️ Hati-hati! Email ini terindikasi berbahaya.", icon="🚨")
                        else:
                            st.balloons() # Efek Balon
                            st.success("✅ KESIMPULAN: HAM (AMAN)")
                            st.toast("✅ Email aman untuk dibuka.", icon="🛡️")
                            
                    except Exception as e:
                        st.error(f"Error prediksi: {e}")
                        st.info("Tips: Pastikan model .pkl berisi Pipeline lengkap.")
                else:
                    st.warning("Mohon ketik isi email terlebih dahulu.")
        else:
            st.error(f"❌ File model `{filename}` tidak ditemukan. Pastikan folder models sudah lengkap.")
