import numpy as np
import pandas as pd
import streamlit as st

# Konfigurasi Halaman
st.set_page_config(
    page_title="Hybrid AI HAR - Kelompok A6",
    page_icon="🏃",
    layout="wide",
)

# Header Utama
st.title("🏃 Intelligent Human Activity Recognition (HAR) System")
st.caption(
    "Tugas Besar Kecerdasan Buatan (TE2514031) | Hybrid Intelligent System: GA + PSO + CNN-LSTM + Fuzzy Logic"
)

# Sidebar Navigasi (8 Menu Wajib Sesuai Panduan)
st.sidebar.header("Navigasi Sistem")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    [
        "1. Project Overview",
        "2. Dataset Explorer (EDA)",
        "3. GA Optimization",
        "4. PSO Optimization",
        "5. Neural Network (CNN-LSTM)",
        "6. Fuzzy Inference System",
        "7. Prediction / Simulation",
        "8. Model Comparison",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Status Progres:** Minggu 12 (EDA, Baseline CNN-LSTM & Kerangka Dashboard)\n\n"
    "**Kelompok A6 - Kelas A**\n"
    "Institut Teknologi Kalimantan"
)

# ==========================================
# HALAMAN 1: PROJECT OVERVIEW
# ==========================================
if menu == "1. Project Overview":
    st.subheader("Gambaran Umum Proyek & Arsitektur Sistem")

    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.markdown(
            """
            **Latar Belakang Masalah:**  
            Pengenalan aktivitas fisik manusia (*Human Activity Recognition*) berbasis sensor *smartphone* (*accelerometer* dan *gyroscope*) memiliki dimensi fitur yang sangat besar (561 fitur). Tanpa penyaringan fitur dan penyetelan arsitektur yang tepat, model prediksi rentan mengalami komputasi lambat dan *overfitting*.

            **Alur Kerja Sistem Hibrida (Pipeline):**
            1. **Dataset UCI ID 240 & Preprocessing:** Normalisasi *StandardScaler* (fit hanya pada data latih untuk mencegah *data leakage*).
            2. **Genetic Algorithm (GA):** Menyeleksi subset fitur sensor paling informatif dari 561 fitur.
            3. **Particle Swarm Optimization (PSO):** Mengoptimasi *hyperparameter* CNN-LSTM (*filter, kernel, LSTM units, dropout, learning rate*).
            4. **CNN-LSTM:** Ekstraksi pola spasial-temporal untuk klasifikasi 6 aktivitas fisik.
            5. **Fuzzy Logic (FIS):** Menentukan tingkat keyakinan deteksi (*Activity / State Confidence*).
            """
        )

    with col2:
        st.markdown("**Identitas Kelompok A6:**")
        anggota_df = pd.DataFrame(
            {
                "NIM": ["04231033", "04231082", "04231052", "04231015"],
                "Nama Mahasiswa": [
                    "Ferry Kurniawan",
                    "Stepanus Pabarrung",
                    "Muhammad Fadhilah Rahman",
                    "Balin Priya Suhada",
                ],
                "Fokus Peran": [
                    "Pipeline & Baseline CNN-LSTM",
                    "Genetic Algorithm (Feature Selection)",
                    "PSO Tuning & Evaluasi Model",
                    "Fuzzy Logic & Streamlit Dashboard",
                ],
            }
        )
        st.dataframe(anggota_df, use_container_width=True, hide_index=True)

        st.markdown(
            """
            - **Mata Kuliah:** Kecerdasan Buatan (TE2514031)  
            - **Dosen Pengampu:** Happy Aprillia, S.ST., M.T., M.Eng., Ph.D.  
            - **Dataset:** UCI Machine Learning Repository (ID 240)
            """
        )

# ==========================================
# HALAMAN 2: DATASET EXPLORER (EDA)
# ==========================================
elif menu == "2. Dataset Explorer (EDA)":
    st.subheader("Eksplorasi Dataset UCI HAR (ID 240)")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Sampel", "10.299 baris")
    m2.metric("Jumlah Fitur Sensor", "561 fitur")
    m3.metric("Missing Values", "0 (Bersih)")
    m4.metric("Rentang Nilai Mentah", "[-1.00, 1.00]")

    st.markdown("### Distribusi Sampel per Kelas Aktivitas")
    eda_df = pd.DataFrame(
        {
            "Aktivitas": [
                "WALKING",
                "WALKING_UPSTAIRS",
                "WALKING_DOWNSTAIRS",
                "SITTING",
                "STANDING",
                "LAYING",
            ],
            "Train (7.352)": [1226, 1073, 986, 1286, 1374, 1407],
            "Test (2.947)": [496, 471, 420, 491, 532, 537],
            "Total": [1722, 1544, 1406, 1777, 1906, 1944],
            "Persentase (%)": [16.72, 14.99, 13.65, 17.25, 18.51, 18.88],
        }
    )

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.dataframe(eda_df, use_container_width=True, hide_index=True)
    with c2:
        chart_df = eda_df.set_index("Aktivitas")[["Train (7.352)", "Test (2.947)"]]
        st.bar_chart(chart_df)

    st.success(
        "Kesimpulan EDA: Seluruh kelas tersebar seimbang (13,65% - 18,88%) dan bebas dari missing values."
    )

# ==========================================
# HALAMAN 3: GA OPTIMIZATION
# ==========================================
elif menu == "3. GA Optimization":
    st.subheader("Genetic Algorithm (GA) - Seleksi Fitur Sensor")
    st.warning("🚧 Tahap Pengembangan Minggu 13: Implementasi seleksi fitur kromosom biner dari 561 fitur sensor.")
    st.markdown(
        """
        **Rancangan Komponen GA:**
        - **Representasi Kromosom:** Vektor biner panjang 561 (`1` = fitur dipilih, `0` = fitur dibuang).
        - **Fungsi Fitness:** Kombinasi akurasi validasi model dan penalti jumlah fitur terpilih.
        - **Output yang Akan Ditampilkan:** Kurva konvergensi fitness per generasi, daftar sensor dominan terpilih, dan perbandingan waktu komputasi.
        """
    )

# ==========================================
# HALAMAN 4: PSO OPTIMIZATION
# ==========================================
elif menu == "4. PSO Optimization":
    st.subheader("Particle Swarm Optimization (PSO) - Hyperparameter Tuning")
    st.warning("🚧 Tahap Pengembangan Minggu 14: Optimasi otomatis hyperparameter arsitektur CNN-LSTM.")
    st.markdown(
        """
        **Ruang Pencarian (Search Space) PSO:**
        - **Conv1D Filters:** `[16, 32, 64, 128]`
        - **Kernel Size:** `[3, 5, 7]`
        - **LSTM Units:** `[32, 64, 128]`
        - **Dropout Rate:** `[0.1 - 0.5]`
        - **Learning Rate:** `[1e-4 - 1e-2]`
        """
    )

# ==========================================
# HALAMAN 5: NEURAL NETWORK (BASELINE)
# ==========================================
elif menu == "5. Neural Network (CNN-LSTM)":
    st.subheader("Arsitektur & Hasil Evaluasi Baseline CNN-LSTM (Minggu 12)")

    col_a, col_b = st.columns([1, 1.3])
    with col_a:
        st.markdown(
            """
            **Arsitektur Baseline:**
            - `Input(shape=(561, 1))`
            - `Conv1D(filters=32, kernel_size=3, activation='relu')`
            - `MaxPooling1D(pool_size=2)`
            - `LSTM(units=64)`
            - `Dropout(0.3)`
            - `Dense(6, activation='softmax')`
            """
        )
        st.metric("Test Accuracy (Baseline)", "81.51%", "2.947 sampel uji")

    with col_b:
        st.markdown("**Classification Report (Data Uji):**")
        report_df = pd.DataFrame(
            {
                "Kelas Aktivitas": [
                    "WALKING",
                    "WALKING_UPSTAIRS",
                    "WALKING_DOWNSTAIRS",
                    "SITTING",
                    "STANDING",
                    "LAYING",
                ],
                "Precision": [0.75, 0.76, 0.71, 0.84, 0.81, 0.99],
                "Recall": [0.66, 0.74, 0.81, 0.82, 0.85, 1.00],
                "F1-Score": [0.70, 0.75, 0.76, 0.83, 0.83, 1.00],
                "Support": [496, 471, 420, 491, 532, 537],
            }
        )
        st.dataframe(report_df, use_container_width=True, hide_index=True)

# ==========================================
# HALAMAN 6: FUZZY INFERENCE SYSTEM
# ==========================================
elif menu == "6. Fuzzy Inference System":
    st.subheader("Fuzzy Logic Layer - Activity / State Confidence")
    st.warning("🚧 Tahap Pengembangan Minggu 15: Integrasi Mamdani FIS untuk mengevaluasi keandalan prediksi.")
    st.markdown(
        """
        **Rancangan Fuzzy Inference System:**
        - **Input Fuzzy:** Probabilitas maksimum kelas dari CNN-LSTM & margin selisih dengan probabilitas tertinggi kedua.
        - **Output Keputusan:** Tingkat keyakinan deteksi (*Low Confidence / Transition*, *Moderate Confidence*, *High Confidence*).
        """
    )

# ==========================================
# HALAMAN 7: PREDICTION / SIMULATION
# ==========================================
elif menu == "7. Prediction / Simulation":
    st.subheader("Simulasi Prediksi Aktivitas & Keputusan Fuzzy")
    st.info("Mode Pratinjau (Minggu 12) — Integrasi penuh unggah file CSV sensor akan aktif pada Minggu 15.")

    prob_sim = st.slider(
        "Simulasi Skor Probabilitas Tertinggi Output CNN-LSTM:",
        min_value=0.17,
        max_value=1.00,
        value=0.89,
        step=0.01,
    )
    if prob_sim >= 0.80:
        st.success(f"Probabilitas: {prob_sim:.2f} → Keputusan Awal: **High Confidence (Aktivitas Valid)**")
    elif prob_sim >= 0.55:
        st.warning(f"Probabilitas: {prob_sim:.2f} → Keputusan Awal: **Moderate Confidence (Perlu Pemantauan)**")
    else:
        st.error(f"Probabilitas: {prob_sim:.2f} → Keputusan Awal: **Low Confidence / Transition State**")

# ==========================================
# HALAMAN 8: MODEL COMPARISON
# ==========================================
elif menu == "8. Model Comparison":
    st.subheader("Tabel Eksperimen Pembanding (Ablation Study)")
    comp_df = pd.DataFrame(
        {
            "Kode": ["Model A", "Model B", "Model C", "Model D", "Model E"],
            "Konfigurasi": [
                "Baseline CNN-LSTM",
                "GA + CNN-LSTM",
                "PSO + CNN-LSTM",
                "GA + PSO + CNN-LSTM",
                "GA + PSO + CNN-LSTM + Fuzzy",
            ],
            "Jumlah Fitur": ["561", "Target M13", "561", "Target M14", "Target M15"],
            "Test Accuracy": ["81.51%", "Menunggu M13", "Menunggu M14", "Menunggu M14", "Sistem Final"],
            "Status": ["Selesai (M12)", "Terjadwal M13", "Terjadwal M14", "Terjadwal M14", "Terjadwal M15"],
        }
    )
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
