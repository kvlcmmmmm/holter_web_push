import pandas as pd
import numpy as np
import biosppy
import neurokit2 as nk
from io import StringIO

class ECGProcessor:

    def read_txt_file_and_extract_first_column(self, file):
        # Dosyayı oku
        with open(file, 'r') as file:
            lines = file.readlines()
        
        # Veriyi bir tek metin dizesine birleştirin
        data_text = ''.join(lines)
        
        # StringIO kullanarak metin veriyi bir dataframe'e dönüştürün
        df = pd.read_csv(StringIO(data_text), sep='\t', skiprows=2)
        print(df)

        # İlk sütunu seçme
        df = df.iloc[:, 1]
    
        return df
    

    def read_csv_file_and_extract_first_column(self, file):
        try:
            df = pd.read_csv(file)
            df = df.iloc[:, 0]
            return df
        except Exception as e:
            print(f"Dosya okunurken bir hata oluştu: {e}")
            return None


    @staticmethod
    def process_ecg_data(df):
        # Process ECG data
        data = biosppy.signals.ecg.ecg(signal=df, sampling_rate=360, show=False)
        r_peaks = data['rpeaks']
        df_filtered = data['filtered']
        df_filtered= nk.rescale(df_filtered, to=[0, 1], scale=None)

        return r_peaks, df_filtered

    @staticmethod
    def window_ecg_signal(ecg_signal, r_peaks, pre_peak=576, post_peak=624):
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

    def predict(self, model, df):
        # Predictions
        X_test = np.array(df).reshape(df.shape[0], df.shape[1], 1)
        predIdxs = model.predict(X_test, batch_size=32)
        predIdxs = np.argmax(predIdxs, axis=1)

        print(df)
        
        return predIdxs



