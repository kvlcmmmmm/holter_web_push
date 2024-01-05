import pandas as pd
import numpy as np
import biosppy
import neurokit2 as nk
import wfdb
from io import StringIO
from scipy.interpolate import interp1d
import os
import statistics
import re

class ECGProcessor:

    def read_txt_file_and_extract_first_column(self, file):
        # Dosyayı oku
        with open(file, 'r') as file:
            lines = file.readlines()
        
        # Veriyi bir tek metin dizesine birleştirin
        data_text = ''.join(lines)
        
        # StringIO kullanarak metin veriyi bir dataframe'e dönüştürün
        df = pd.read_csv(StringIO(data_text), sep='\t', skiprows=2)
        df = df.iloc[:, 1]

        print(df)

        return df
    

    def read_csv_file_and_extract_first_column(self, file):
        try:
            df = pd.read_csv(file)
            df = df.iloc[:, 0]
            return df
        except Exception as e:
            print(f"Dosya okunurken bir hata oluştu: {e}")
            return None
        
    def read_csv_file_and_extract_first_samples(self, file):
        try:
            df = pd.read_csv(file,  sep=',', skiprows=1, header=None)
            nonbeat = ['[', '!', ']', 'x', '(', ')', 'p', 't', 'u', '`', "'", '^', '~', '+', 's', 'T', '*', 'D', '=', '"', '@']
            # Regex deseni oluşturma
            regex_pattern = '|'.join(re.escape(char) for char in nonbeat)
            # Silinmesi istenen karakterleri içeren satırları tespit etme
            mask = df.apply(lambda row: row.astype(str).str.contains(regex_pattern, regex=True).any(), axis=1)
            # Bu satırları DataFrame'den çıkarma
            df = df[~mask]
            df = df[0]
            print(df)
            return df
        except Exception as e:
            print(f"Dosya okunurken bir hata oluştu: {e}")
            return None
        

    def read_wfdb_file_and_extract_first_column(self, file_dat):
        try:
            # .dat dosyasının adını al ve uzantısını kaldır
            record_name = os.path.splitext(file_dat)[0]

            # WFDB dosyasını oku
            df = wfdb.rdrecord(record_name)

            # İlk sütunu döndür
            return df.p_signal[:, 0]
        except Exception as e:
            print(f"Dosya okunurken bir hata oluştu: {e}")
            return None


    def signal_resample(self, df, frequency):
        fs_original = frequency  # tahmin edilecek verinin örnekleme frekansı
        fs_new = 360  # yeni örnekleme frekansı (Model 360 Hz data ile eğitildiği için.)

        ekg_original = df  #EKG sinyali

        # Orijinal sinyal dizinleri (indeksler)
        indices_original = np.arange(len(ekg_original))

        # Interpolasyon faktörünü hesaplama
        interpolation_factor = fs_new / fs_original

        # Yeni interpolasyon için dizinler
        indices_new = np.linspace(0, len(ekg_original) - 1, int(len(ekg_original) * interpolation_factor))

        # Lineer interpolasyon uygulama
        interpolator = interp1d(indices_original, ekg_original, kind='linear')
        ekg_interpolated_without_time = interpolator(indices_new)

        df = ekg_interpolated_without_time

        return df


    @staticmethod
    def process_ecg_data(df):
        # Process ECG data
        data = biosppy.signals.ecg.ecg(signal=df, sampling_rate=360, show=False)
        r_peaks = data['rpeaks']
        df_filtered = data["filtered"]
        df_filtered_draw = data["filtered"]
        df_filtered_draw= nk.rescale(df_filtered_draw, to=[0, 1], scale=None)

        return r_peaks, df_filtered, df_filtered_draw


    @staticmethod
    def window_ecg_signal(ecg_signal, r_peaks, pre_peak=144, post_peak=216):
        # Windowing function
        """
        EKG sinyalini R-pik noktaları baz alınarak belirli bir pencere boyutunda böler.

        Parametreler:
        ecg_signal: EKG sinyali, bir numpy dizisi veya liste olmalıdır.
        r_peaks: R-pik noktalarının zaman indeksleri.
        pre_peak: R-pik noktasından önceki pencere boyutu.
        post_peak: R-pik noktasından sonraki pencere boyutu.

        Döndürür:
        windows: EKG sinyalinin bölünmüş pencerelerinin listesi.
        """

        windows = []
        r_peaks_new = []
        for r_peak in r_peaks:
            start = r_peak - pre_peak
            end = r_peak + post_peak

            # Pencere, sinyalin başlangıcı veya sonunu aşmamalıdır
            if start >= 0 and end <= len(ecg_signal):
                window = ecg_signal[start:end]

                windows.append(window)
                r_peaks_new.append(r_peak)

        return windows, r_peaks_new


    @staticmethod
    def create_dataframe_from_windows_fast(windows):
        # DataFrame creation function
        """
        Pencereleri hızlı bir şekilde DataFrame'e dönüştürür.

        Parametreler:
        windows: EKG sinyalinin bölünmüş pencerelerinin listesi.

        Döndürür:
        df: Oluşturulan DataFrame.
        """
        # Pencereleri bir NumPy dizisine dönüştür
        array = np.array(windows)

        # NumPy dizisini kullanarak DataFrame oluştur
        df = pd.DataFrame(array)

        return df
        
        
    @staticmethod
    def normalize_rows(df):
        # Her satırın min ve max değerlerini hesapla
        min_values = df.min(axis=1)
        max_values = df.max(axis=1)

        # Her satırı kendi min ve max değerlerine göre normalize et
        normalized_df = (df.sub(min_values, axis=0)).div(max_values - min_values, axis=0)
        return normalized_df

    def predict(self, model, df):
        # Predictions
        X_test = np.array(df).reshape(df.shape[0], df.shape[1], 1)
        predIdxs = model.predict(X_test, batch_size=32)
        predIdxs = np.argmax(predIdxs, axis=1)

        print(df)
        
        return predIdxs



