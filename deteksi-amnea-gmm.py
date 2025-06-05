import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.signal import welch
import time

st.title("Implementasi Algoritma GMM untuk Deteksi Pola Tidur melalui Sinyal Electroencephalography (EEG)")

# DOWNLOAD FILE CSV
file_path = "extrait_wSleepPage01.csv"

@st.cache_data
def load_file(file_path):
    try:
        with open(file_path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        st.warning(f"File {file_path} tidak ditemukan. Silakan unggah file CSV Anda sendiri.")
        return None

csv_data = load_file(file_path)
if csv_data is not None:
    st.download_button(
        label="Download extrait_wSleepPage01.csv",
        data=csv_data,
        file_name="extrait_wSleepPage01.csv",
        mime="text/csv")

# INPUT FILE CSV
def upload_csv_file():
    uploaded_file = st.file_uploader("Unggah file CSV Sinyal EEG:", type=["csv"])
    if uploaded_file is not None:
        try:
            # Coba dengan berbagai format yang mungkin
            try:
                df = pd.read_csv(uploaded_file, sep=";", decimal=",")
            except:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file)
                except:
                    # Jika masih gagal, coba dengan opsi lain
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=None, engine='python')
            
            st.write("Data CSV Sinyal EEG:")
            st.dataframe(df)
            st.write("Statistik Deskriptif:")
            st.dataframe(df.describe())
            return df
        except Exception as e:
            st.error(f"Error saat membaca file: {e}")
            return None
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

    possible_signal_cols = [col for col in df.columns if any(x in col.upper() for x in ['EEG', 'EOG'])]
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

    # FITUR YANG TERPILIH
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

        if time_col:
            max_time = float(df[time_col].max())
            time_range = st.slider("Pilih rentang waktu (detik):", 
                                  0.0, max_time, 
                                  (0.0, min(10.0, max_time)))
            
            filtered_df = df[(df[time_col] >= time_range[0]) & (df[time_col] <= time_range[1])]
        else:
            sample_range = st.slider("Pilih rentang sampel:", 
                                    0, len(df), 
                                    (0, min(2000, len(df))))
            filtered_df = df.iloc[sample_range[0]:sample_range[1]]

        if len(filtered_df) > 0:
            plt.figure(clear=True)
            
            # Buat subplot untuk setiap fitur terpilih
            n_features = len(st.session_state['selected_features'])
            fig, axes = plt.subplots(n_features, 1, figsize=(10, 3 * n_features), clear=True)
            
            if n_features == 1:
                axes = [axes]
            
            # Plot setiap fitur di subplot terpisah
            for i, feature in enumerate(st.session_state['selected_features']):
                if feature in filtered_df.columns:
                    # Bersihkan subplot
                    axes[i].clear()
                    
                    # Plot data
                    if time_col and time_col in filtered_df.columns:
                        axes[i].plot(filtered_df[time_col], filtered_df[feature], linewidth=1)
                        axes[i].set_xlabel('Waktu (detik)')
                    else:
                        axes[i].plot(filtered_df.index, filtered_df[feature], linewidth=1)
                        axes[i].set_xlabel('Sampel')
                    
                    # Atur label dan judul
                    axes[i].set_ylabel(feature)
                    axes[i].set_title(f'Sinyal {feature}')
                    axes[i].grid(True)
            
            # Atur layout dan tampilkan
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)  # Tutup figure untuk mencegah penumpukan

        if st.button("Simpan Fitur Terpilih"):
            st.success(f"Berhasil memilih {len(st.session_state['selected_features'])} fitur!")
            st.session_state['channels'] = st.session_state['selected_features']
            
            # Tentukan window_self.size default (30 detik pada 200 Hz)
            fs = 200  # Frekuensi sampling default
            window_seconds = 30
            window_size = window_seconds * fs
            st.session_state['window_self.size'] = window_size
            
            st.subheader("Inisialisasi Fitur untuk GMM")
            return st.session_state['selected_features']
    else:
        st.info("Belum ada fitur yang dipilih. Silakan pilih minimal satu fitur.")
        return None

# GAUSSIAN MIXTURE MODEL
class GMM():
    def __init__(self, n_components=5, max_iter=100, tol=1e-4, random_state=None, reg_covar=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.mu = None      
        self.sigma = None   
        self.pi = None      
        self.X_mean = None
        self.X_std = None
        self.log_likelihoods = []
   
        if random_state is not None:
            np.random.seed(random_state)
    
    def gaussian_pdf(self, x, mean, cov):
        self.size = len(x)
        self.det = np.linalg.det(cov)
        if self.det <= 0:  
            self.det = 1e-6
        norm_const = 1.0 / (np.power((2*np.pi), self.size/2) * np.sqrt(self.det))
        x_mu = x - mean

        try:
            inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_reg = cov + np.eye(self.size) * self.reg_covar
            inv = np.linalg.inv(cov_reg)
        
        result = np.exp(-0.5 * np.dot(np.dot(x_mu, inv), x_mu))
        return norm_const * result
    
    def initialize_parameters(self, X):
        n, d = X.shape
        indices = np.random.choice(n, 1)
        self.mu = X[indices].copy()

        for i in range(1, self.n_components):
            min_sq_dist = np.min(np.sum((X[:, np.newaxis, :] - self.mu[np.newaxis, :, :]) ** 2, axis=2), axis=1)
            probs = min_sq_dist / np.sum(min_sq_dist)
            next_idx = np.random.choice(n, 1, p=probs)
            self.mu = np.vstack([self.mu, X[next_idx]])
        
        self.sigma = np.array([np.cov(X.T) + np.eye(d)*self.reg_covar for _ in range(self.n_components)])
        self.pi = np.ones(self.n_components) / self.n_components
    
    def e_step(self, X):
        n = X.shape[0]
        gamma = np.zeros((n, self.n_components))
        
        for i in range(self.n_components):
            for j in range(n):
                gamma[j, i] = self.pi[i] * self.gaussian_pdf(X[j], self.mu[i], self.sigma[i])
        
        gamma_sum = np.sum(gamma, axis=1)[:, np.newaxis]
        gamma_sum[gamma_sum < 1e-10] = 1e-10 
        gamma /= gamma_sum
        
        return gamma
    
    def m_step(self, X, gamma):
        n, d = X.shape
        N_k = np.sum(gamma, axis=0)
        N_k[N_k < 1e-10] = 1e-10 
        self.mu = np.dot(gamma.T, X) / N_k[:, np.newaxis]
        
        self.sigma = np.zeros((self.n_components, d, d))
        for i in range(self.n_components):
            x_mu = X - self.mu[i]
            weighted = gamma[:, i][:, np.newaxis] * x_mu
            self.sigma[i] = np.dot(weighted.T, x_mu) / N_k[i]
            self.sigma[i] += np.eye(d) * self.reg_covar

        self.pi = N_k / n
    
    def compute_log_likelihood(self, X):
        n = X.shape[0]
        likelihood = np.zeros((n, self.n_components))
        
        for i in range(self.n_components):
            for j in range(n):
                likelihood[j, i] = self.pi[i] * self.gaussian_pdf(X[j], self.mu[i], self.sigma[i])
        
        total = np.sum(likelihood, axis=1)
        total[total < 1e-10] = 1e-10 
        log_likelihood = np.sum(np.log(total))
        
        return log_likelihood
    
    def fit(self, X, standardize=True):
        if isinstance(X, pd.DataFrame):
            X = X.values

        if standardize:
            self.X_mean = X.mean(axis=0)
            self.X_std = X.std(axis=0)
            self.X_std[self.X_std < 1e-10] = 1.0
            X = (X - self.X_mean) / self.X_std

        self.initialize_parameters(X)
        self.log_likelihoods = []

        log_placeholder = st.empty()
        progress_bar = st.progress(0)

        for iteration in range(self.max_iter):
            start_time = time.time()

            gamma = self.e_step(X)
            self.m_step(X, gamma)

            ll = self.compute_log_likelihood(X)
            self.log_likelihoods.append(ll)

            end_time = time.time()

            log_text = f"Iterasi {iteration + 1}/{self.max_iter}, Log-likelihood: {ll:.6f}, Waktu: {end_time - start_time:.2f} detik"
            log_placeholder.text(log_text)
            progress_bar.progress((iteration + 1) / self.max_iter)

            if iteration > 0 and abs(self.log_likelihoods[-1] - self.log_likelihoods[-2]) < self.tol:
                log_placeholder.text(
                    log_text + f"\nKonvergen pada iterasi {iteration + 1} dengan perubahan log-likelihood {abs(self.log_likelihoods[-1] - self.log_likelihoods[-2]):.6f}"
                )
                break

        return self

    
    def predict_proba(self, X, standardize=True):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if standardize and self.X_mean is not None and self.X_std is not None:
            X = (X - self.X_mean) / self.X_std
        
        gamma = self.e_step(X)
        return gamma
    
    def predict(self, X, standardize=True):
        probas = self.predict_proba(X, standardize)
        return np.argmax(probas, axis=1)
    
    def compute_bic(self, X):
        n, d = X.shape

        num_params = (self.n_components - 1) + \
                    (self.n_components * d) + \
                    (self.n_components * d * (d + 1) / 2)
        
        bic = -2 * self.log_likelihoods[-1] + num_params * np.log(n)
        return bic
    
    def plot_convergence(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, len(self.log_likelihoods) + 1), self.log_likelihoods, 'o-')
        ax.set_title('Kurva Konvergensi GMM')
        ax.set_xlabel('Iterasi')
        ax.set_ylabel('Log-likelihood')
        ax.grid(True)
        return fig
    
    def plot_clusters_2d(self, X, labels=None, standardize=True, feature_names=None):
        if isinstance(X, pd.DataFrame):
            if feature_names is None and X.columns is not None:
                feature_names = X.columns.tolist()
            X = X.values
        
        if standardize and self.X_mean is not None and self.X_std is not None:
            X = (X - self.X_mean) / self.X_std
        if labels is None:
            labels = self.predict(X, standardize=False) 
        n_samples, n_features = X.shape
        
        if n_features > 2:
            cov_matrix = np.cov(X.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            components = eigenvectors[:, :2]
            
            X_2d = X.dot(components)
            
            if feature_names is None:
                feature_names = ['Principal Component 1', 'Principal Component 2']
            else:
                feature_names = ['PC1', 'PC2']
        else:
            X_2d = X
            if feature_names is None:
                feature_names = ['Feature 1', 'Feature 2']
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
        if n_features > 2:
            means_2d = self.mu.dot(components)
        else:
            means_2d = self.mu
        
        ax.scatter(means_2d[:, 0], means_2d[:, 1], c='red', marker='x', s=100, linewidths=3)
        for k in range(self.n_components):
            if n_features > 2:
                covar_k = components.T.dot(self.sigma[k]).dot(components)
            else:
                covar_k = self.sigma[k]
            
            v, w = np.linalg.eigh(covar_k)
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[1], u[0])
            angle = 180.0 * angle / np.pi  
            ell = plt.matplotlib.patches.Ellipse(
                means_2d[k], v[0], v[1], angle=180.0 + angle, 
                edgecolor='black', facecolor='none', linewidth=2)
            ax.add_patch(ell)
        
        plt.colorbar(scatter, label='Cluster')
        ax.set_title('Hasil Clustering GMM')
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.grid(True)
        
        return fig

# FITUR ENGINEERING
def extract_frequency_features(df, channels, fs=200):
    features = []
    
    freq_bands = {
        'Delta': (0.5, 4),   # N3
        'Theta': (4, 8),     # N1 dan N2
        'Alpha': (8, 12),    # W
        'Sigma': (12, 16),   # N2
        'Beta': (16, 30)     # W dan REM
    }
    
    for ch in channels:
        if ch in df.columns:
            if df[ch].dtype == 'object':
                sig = df[ch].replace(',', '.', regex=True).astype(float).values
            else:
                sig = df[ch].values
            
            if len(sig) >= fs*2:
                f, psd = welch(sig, fs=fs, nperseg=fs*2)
            elif len(sig) > 0:
                nperseg = min(len(sig), 256)
                f, psd = welch(sig, fs=fs, nperseg=nperseg)
            else:
                features.extend([0.0] * len(freq_bands))
                continue
            for band_name, (fmin, fmax) in freq_bands.items():
                idx = np.logical_and(f >= fmin, f <= fmax)
                if np.sum(idx) > 0:
                    power = np.trapz(psd[idx], f[idx])
                else:
                    power = 0.0
                features.append(power)
        else:
            features.extend([0.0] * len(freq_bands))
    
    return features

#HITUNG APNEA
def is_apnea(prob_dict, rem_thresh=30, n1_thresh=15):
    if prob_dict['R'] >= rem_thresh or prob_dict['N1'] >= n1_thresh:
        return 1
    return 0

cluster_names = ['W(%)', 'R(%)', 'N1(%)', 'N2(%)', 'N3(%)']

def simple_feature_extraction(df, channels):
    features = []
    for ch in channels:
        if ch in df.columns:
            sig = df[ch].replace(',', '.', regex=True).astype(float).values
            features.append(np.mean(sig))
        else:
            features.append(0.0)
    return features

def label_tahapan_tidur(df):
    apnea_labels = []
    for _, row in df.iterrows():
        prob_dict = {key.strip('%'): row.get(key, 0.0) for key in ['W(%)', 'N1(%)', 'N2(%)', 'N3(%)', 'R(%)']}
        label = is_apnea(prob_dict)
        apnea_labels.append(label)
    df['Apnea'] = apnea_labels
    return df


#VISUALISASI HASIL LABEL PADA DATASET
def menampilkan_data_hasil_klaster(df, channels, window_size=1000, cluster_stage_labels=None):
    if cluster_stage_labels is None:
        cluster_stage_labels = ['W', 'N1', 'N2', 'N3', 'R']
    
    X_features = []
    window_indices = []
    n_samples = len(df)
    
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        use_streamlit = True
    except:
        use_streamlit = False
    
    for i, start in enumerate(range(0, n_samples, window_size)):
        end = min(start + window_size, n_samples)
        window = df.iloc[start:end]
        
        if len(window) < 200:
            continue
        
        window_indices.append((start, end))
        feat = simple_feature_extraction(window, channels)
        X_features.append(feat)
        
        if use_streamlit:
            progress = (i + 1) / ((n_samples // window_size) + 1)
            progress_bar.progress(progress)
            status_text.text(f"Mengekstrak fitur: {i+1}/{(n_samples // window_size) + 1} window")
    
    X = np.array(X_features)
    
    if use_streamlit:
        status_text.text("Melatih model GMM...")
    
    gmm = GMM(n_components=len(cluster_stage_labels), max_iter=100, tol=1e-4)
    gmm.fit(X)
    
    probabilities = gmm.predict_proba(X)
    probabilities_percent = probabilities * 100
    
    for label in cluster_stage_labels:
        df[label + '(%)'] = np.nan
    
    for idx, (start, end) in enumerate(window_indices):
        if idx < len(probabilities_percent):
            for i, label in enumerate(cluster_stage_labels):
                df.loc[start:end-1, label + '(%)'] = probabilities_percent[idx, i] 
    bic = gmm.compute_bic(X)
    
    if use_streamlit:
        status_text.text("Menampilkan hasil...")
        st.subheader("Bayesian Information Criterion (BIC)")
        st.write(f"BIC: {bic:.4f}")
        
        st.subheader("Hasil Probabilitas Tahapan Tidur")
        st.write("5 baris pertama:")
        time_col = 'Time (s)' if 'Time (s)' in df.columns else 'SS' if 'SS' in df.columns else None
        if time_col:
            st.dataframe(df[[time_col] + [label+'(%)' for label in cluster_stage_labels]].head())
        else:
            st.dataframe(df[[label+'(%)' for label in cluster_stage_labels]].head())
    else:
        st.write(f"BIC: {bic:.4f}")
        gmm.plot_convergence()
        gmm.plot_clusters_2d(X)
        st.write("Hasil Probabilitas Tahapan Tidur (5 baris pertama):")
        time_col = 'Time (s)' if 'Time (s)' in df.columns else 'SS' if 'SS' in df.columns else None
        if time_col:
            st.write(df[[time_col] + [label+'(%)' for label in cluster_stage_labels]].head())
        else:
            st.write(df[[label+'(%)' for label in cluster_stage_labels]].head())
    return df, gmm

# TAMPILAN WIDGETS
def tampilan_widget():
    df = upload_csv_file()
    if df is not None:
        # 1. Pengambilan Fitur Data
        selected_features = pilih_fitur(df)
        if selected_features:
            X = df[selected_features].copy()
            gmm = GMM(n_components=5,max_iter=3, random_state=42)
            gmm.fit(X)
            labels = gmm.predict(X)

            # 2. Penampilan Hasil Plot Klaster
            st.success("Clustering GMM selesai.")
            st.write("Label Klaster yang Diprediksi:")
            df_result = df.iloc[:len(labels)].copy()
            df_result['Predicted_Label'] = labels
            st.dataframe(df_result.head(10))

            fig = gmm.plot_clusters_2d(X, labels, feature_names=selected_features)
            st.pyplot(fig)

            fig_ll = gmm.plot_convergence()
            st.pyplot(fig_ll)

if __name__ == "__main__":
    tampilan_widget()
