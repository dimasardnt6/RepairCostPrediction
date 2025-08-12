import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBRegressor
from datetime import datetime

st.set_page_config(page_title="Prediksi Biaya Armada", layout="wide")
st.title("🚍 Prediksi Biaya Perbaikan Armada per Segmen")

# === Load pipeline dan model dari file ===
pipeline = joblib.load("preprocess_pipeline.pkl")
model = joblib.load("xgb_model.pkl")

# === Upload File ===
uploaded_file = st.file_uploader("Unggah file Excel", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Bersihkan dan ubah format
    df['total_biaya'] = df['total_biaya'].astype(str).str.replace(',', '.')
    df['total_biaya'] = pd.to_numeric(df['total_biaya'], errors='coerce')
    df['periode'] = pd.to_datetime(df['bulan'].astype(str).str.zfill(2) + '-' + df['tahun'].astype(str), format='%m-%Y')

    # Identifikasi bulan berikutnya
    last_period = df['periode'].max()
    last_month = last_period.month
    last_year = last_period.year
    next_month = 1 if last_month == 12 else last_month + 1
    next_year = last_year + 1 if last_month == 12 else last_year

    # Buat data input untuk bulan depan
    segment_group = df[df['bulan'] == last_month].groupby('nm_segment').agg({
        'umur_rata2_armada': 'mean',
        'jumlah_total_per_item': 'mean',
        'frekuensi_perbaikan': 'mean'
    }).reset_index()

    segment_group['bulan'] = next_month
    segment_group['tahun'] = next_year
    segment_group['hari_besar'] = 0  # asumsi

    segment_group = segment_group[['nm_segment', 'bulan', 'tahun', 'hari_besar',
                                   'umur_rata2_armada', 'jumlah_total_per_item', 'frekuensi_perbaikan']]

    # Transformasi input
    X_pred = pipeline.transform(segment_group)

    # Prediksi
    segment_group['prediksi_biaya'] = model.predict(X_pred)
    segment_group['prediksi_biaya_rp'] = segment_group['prediksi_biaya'].apply(lambda x: f"Rp {x:,.0f}")

    # Tampilkan hasil
    st.subheader("📊 Prediksi Biaya Bulan Depan")
    st.dataframe(segment_group[['nm_segment', 'bulan', 'tahun', 'prediksi_biaya_rp']])

    # Visualisasi interaktif Streamlit
    st.subheader("📈 Visualisasi Prediksi (Interaktif)")
    chart_data = segment_group[['nm_segment', 'prediksi_biaya']].sort_values(by='prediksi_biaya', ascending=False)
    chart_data.set_index('nm_segment', inplace=True)
    st.bar_chart(chart_data)

    # Tombol Download
    hasil_excel = segment_group[['nm_segment', 'bulan', 'tahun', 'prediksi_biaya']]
    hasil_excel.to_excel("hasil_prediksi.xlsx", index=False)
    with open("hasil_prediksi.xlsx", "rb") as f:
        st.download_button("📥 Unduh Hasil ke Excel", f, file_name="hasil_prediksi.xlsx")
