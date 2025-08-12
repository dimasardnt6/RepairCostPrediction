import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import os

# === Tentukan path absolut file model & encoder ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model_rf.pkl")
encoder_path = os.path.join(BASE_DIR, "label_encoder.pkl")

# === Load model dan encoder dengan pengecekan ===
if not os.path.exists(model_path) or not os.path.exists(encoder_path):
    st.error("❌ File model atau encoder tidak ditemukan. Pastikan `model_rf.pkl` dan `label_encoder.pkl` ada di folder yang sama dengan `app.py`.")
    st.stop()

model = joblib.load(model_path)
le = joblib.load(encoder_path)

st.title("📈 Prediksi Biaya Perbaikan Armada - Bulanan")

# === Upload File ===
uploaded_file = st.file_uploader("📂 Unggah file Excel agregasi historis:", type=[".xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df['segment_encoded'] = le.transform(df['nm_segment'])

    st.subheader("🔍 Data Historis Agregasi")
    st.dataframe(df)

    # Tentukan bulan depan
    last_row = df.sort_values(by=['tahun', 'bulan']).iloc[-1]
    last_bulan, last_tahun = int(last_row['bulan']), int(last_row['tahun'])
    next_bulan = 1 if last_bulan == 12 else last_bulan + 1
    next_tahun = last_tahun + 1 if last_bulan == 12 else last_tahun

    # Hari besar input
    st.subheader("📅 Set Hari Besar untuk Bulan Prediksi")
    hari_besar_option = st.radio(
        f"Apakah bulan {next_bulan} {next_tahun} termasuk hari besar?",
        ('Tidak', 'Ya'),
        key="radio_haribesar"
    )
    hari_besar_flag = 1 if hari_besar_option == 'Ya' else 0

    # Tombol Prediksi
    if st.button("🚀 Prediksi Bulan Depan"):
        agg_by_segment = df.groupby('nm_segment').agg({
            'segment_encoded': 'first',
            'umur_rata2_armada': 'mean',
            'jumlah_total_per_item': 'mean',
            'frekuensi_perbaikan': 'mean'
        }).reset_index()

        agg_by_segment['bulan'] = next_bulan
        agg_by_segment['tahun'] = next_tahun
        agg_by_segment['hari_besar'] = hari_besar_flag

        X_pred = agg_by_segment[['segment_encoded', 'bulan', 'tahun', 'hari_besar',
                                 'umur_rata2_armada', 'jumlah_total_per_item', 'frekuensi_perbaikan']]
        agg_by_segment['prediksi_total_biaya'] = model.predict(X_pred)

        st.session_state.prediksi_df = agg_by_segment.copy()

    # Jika prediksi sudah dilakukan
    if 'prediksi_df' in st.session_state:
        pred_df = st.session_state.prediksi_df

        st.subheader(f"📊 Hasil Prediksi Biaya Bulan {next_bulan} {next_tahun}")
        st.dataframe(pred_df[['nm_segment', 'bulan', 'tahun', 'hari_besar', 'prediksi_total_biaya']])

        # Gabungkan dengan historis
        df_hist = df.copy()
        df_hist['periode'] = pd.to_datetime(dict(year=df_hist['tahun'], month=df_hist['bulan'], day=1))
        df_hist['total_biaya'] = df_hist['total_biaya']
        df_hist['jenis'] = 'History'

        pred_plot = pred_df.copy()
        pred_plot['periode'] = pd.to_datetime(dict(year=pred_plot['tahun'], month=pred_plot['bulan'], day=1))
        pred_plot['total_biaya'] = pred_plot['prediksi_total_biaya']
        pred_plot['jenis'] = 'Prediksi'

        gabung = pd.concat([
            df_hist[['nm_segment', 'periode', 'total_biaya', 'jenis']],
            pred_plot[['nm_segment', 'periode', 'total_biaya', 'jenis']]
        ], ignore_index=True)

        # === Segment Filter ===
        st.subheader("📈 Visualisasi Biaya Per Bulan per Segment")
        unique_segments = gabung['nm_segment'].unique()
        selected_segment = st.selectbox("Pilih Segment:", unique_segments)

        df_filtered = gabung[gabung['nm_segment'] == selected_segment].sort_values('periode')

        # === Grafik Tersambung + Titik Prediksi ===
        fig = go.Figure()

        # Garis utama
        fig.add_trace(go.Scatter(
            x=df_filtered['periode'],
            y=df_filtered['total_biaya'],
            mode='lines+markers',
            name='Total Biaya',
            line=dict(color='blue'),
            marker=dict(color='blue')
        ))

        # Titik prediksi merah
        pred_point = df_filtered[df_filtered['jenis'] == 'Prediksi']
        fig.add_trace(go.Scatter(
            x=pred_point['periode'],
            y=pred_point['total_biaya'],
            mode='markers',
            name='Titik Prediksi',
            marker=dict(color='red', size=12, symbol='circle'),
            showlegend=True
        ))

        fig.update_layout(
            title=f"Biaya Perbaikan Bulanan - Segment: {selected_segment}",
            xaxis_title="Periode",
            yaxis_title="Total Biaya",
            legend_title="Keterangan",
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)
