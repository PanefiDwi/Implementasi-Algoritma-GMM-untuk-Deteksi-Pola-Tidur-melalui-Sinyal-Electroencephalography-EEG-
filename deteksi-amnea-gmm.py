import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title("Implementasi Algoritma GMM untuk Deteksi Pola Tidur melalui Sinyal Electroencephalography (EEG)")

# DOWNLOAD FILE CSV
file_path = "extrait_wSleepPage01.csv"

@st.cache_data
def load_file(file_path):
    with open(file_path, "rb") as f:
        return f.read()

csv = load_file(file_path)

st.download_button(
    label="Download extrait_wSleepPage01.csv",
    data=csv,
    file_name="extrait_wSleepPage01.csv",
    mime="text/csv")

# INPUT FILE CSV
def upload_csv_file():
    uploaded_file = st.file_uploader("Unggah file CSV Sinyal EEG:", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, sep=";", decimal=",")
        st.write("Data CSV Sinyal EEG:")
        st.dataframe(df)
        st.dataframe(df.describe())
        return df
    else:
        st.warning("Silakan unggah file CSV terlebih dahulu.")
        return None

# FUNGSI PEMILIHAN FITUR
def pilih_fitur(df):
    if df is None:
        st.warning("Tidak ada data untuk memilih fitur.")
        return None

    st.write("## Pemilihan Fitur untuk Analisis GMM")
    st.markdown("""
    Pemilihan fitur yang tepat sangat penting untuk keberhasilan model GMM. Dalam konteks analisis pola tidur 
    melalui EEG, fitur-fitur yang relevan mencakup sinyal dari berbagai channel dan fitur turunan.
    """)

    if 'selected_features' not in st.session_state:
        st.session_state['selected_features'] = []

    possible_signal_cols = [col for col in df.columns if any(x in col for x in ['EEG', 'EOG'])]
    other_cols = [col for col in df.columns if col not in possible_signal_cols]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Sinyal EEG dan EOG:**")
        for col in possible_signal_cols:
            if st.checkbox(col, value=col in st.session_state['selected_features'], key=f"check_{col}"):
                if col not in st.session_state['selected_features']:
                    st.session_state['selected_features'].append(col)
            else:
                if col in st.session_state['selected_features']:
                    st.session_state['selected_features'].remove(col)

    with col2:
        st.markdown("**Kolom Lainnya:**")
        for col in other_cols:
            if st.checkbox(col, value=col in st.session_state['selected_features'], key=f"check_{col}"):
                if col not in st.session_state['selected_features']:
                    st.session_state['selected_features'].append(col)
            else:
                if col in st.session_state['selected_features']:
                    st.session_state['selected_features'].remove(col)
#FITUR YANG TERPILIH
    st.subheader("Fitur yang Terpilih")
    if st.session_state['selected_features']:
        selected_features_df = pd.DataFrame({
            'No': range(1, len(st.session_state['selected_features']) + 1),
            'Nama Fitur': st.session_state['selected_features']
        })
        st.table(selected_features_df)

        st.subheader("Visualisasi Fitur Terpilih")
        time_col = None
        if 'Time (s)' in df.columns:
            time_col = 'Time (s)'
        # elif 'SS' in df.columns:
        #     time_col = 'SS'

        if time_col:
            max_time = df[time_col].max()
            time_range = st.slider("Pilih rentang waktu (detik):", 0.0, float(max_time),
                                   (0.0, min(10.0, float(max_time))))
            filtered_df = df[(df[time_col] >= time_range[0]) & (df[time_col] <= time_range[1])]
        else:
            sample_range = st.slider("Pilih rentang sampel:", 0, len(df),
                                     (0, min(2000, len(df))))
            filtered_df = df.iloc[sample_range[0]:sample_range[1]]

        if len(filtered_df) > 0:
            fig, ax = plt.subplots(len(st.session_state['selected_features']), 1,
                                   figsize=(10, 2 * len(st.session_state['selected_features'])))
            if len(st.session_state['selected_features']) == 1:
                ax = [ax]

            for i, feature in enumerate(st.session_state['selected_features']):
                if feature in filtered_df.columns:
                    if time_col:
                        ax[i].plot(filtered_df[time_col], filtered_df[feature])
                        ax[i].set_xlabel('Waktu (detik)')
                    else:
                        ax[i].plot(filtered_df[feature])
                        ax[i].set_xlabel('Sampel')
                    ax[i].set_ylabel(feature)
                    ax[i].set_title(f'Sinyal {feature}')
                    ax[i].grid(True)

            plt.tight_layout()
            st.pyplot(fig)


        if st.button("Simpan Fitur Terpilih"):
            st.success(f"Berhasil memilih {len(st.session_state['selected_features'])} fitur!")
            st.subheader("Inisialisasi Fitur untuk GMM")
            st.code(
                "# Inisialisasi fitur untuk GMM\n"
                "selected_features = {}\n\n"
                "# Ekstraksi fitur dari data\n"
                "features_data = []\n"
                "for feature in selected_features:\n"
                "    feature_values = df[feature].values\n"
                "    features_data.append(feature_values)\n"
                "    print(f\"Fitur: {feature}\")\n"
                "    print(f\"- Jumlah sampel: {len(feature_values)}\")\n"
                "    print(f\"- Nilai min: {feature_values.min():.4f}\")\n"
                "    print(f\"- Nilai max: {feature_values.max():.4f}\")\n"
                "    print(f\"- Rata-rata: {feature_values.mean():.4f}\")\n"
                "    print(f\"- Std deviasi: {feature_values.std():.4f}\")\n"
                "    print(\"-\" * 50)\n\n"
                "# Siapkan data untuk GMM\n"
                "X = np.column_stack(features_data)\n"
                "print(f\"Data siap untuk GMM dengan dimensi: {X.shape}\")"
            )
            return st.session_state['selected_features']
    else:
        st.info("Belum ada fitur yang dipilih. Silakan pilih minimal satu fitur.")
        return None

# TAMPILAN WIDGETS
def tampilan_widget():
    df = upload_csv_file()
    if df is not None:
        pilih_fitur(df)

tampilan_widget()
