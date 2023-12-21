from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import tempfile
import keras
import uuid
from services import ECGProcessor

ecg_processor = ECGProcessor()

# Modelin yüklenmesi ve hazırlanması
model_path = "model_12122023.h5"

model = keras.models.load_model(model_path, compile=False)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
temp_directory = tempfile.gettempdir()

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        frequency = int(request.values['frequency'])
        if file.filename == '':
            return "No selected file", 400
        if frequency == '':
            return "No frequency", 400
        if file:
            filename = secure_filename(str(uuid.uuid4()) + file.filename)
            file_path = os.path.join(temp_directory, filename)
            file.save(file_path)
            print(file_path)

            # Dosya işleme
            try:
                # Dosya uzantısına göre işlem yapma
                if file_path.endswith('.txt'):
                    df = ecg_processor.read_txt_file_and_extract_first_column(file_path)
                    df = ecg_processor.signal_resample(df, frequency)
                elif file_path.endswith('.csv'):
                    df = ecg_processor.read_csv_file_and_extract_first_column(file_path)
                    df = ecg_processor.signal_resample(df, frequency)
                else:
                    raise ValueError("Unsupported file type")

                r_peaks, df_filtered = ecg_processor.process_ecg_data(df)
                windows, r_peaks_new = ecg_processor.window_ecg_signal(df_filtered, r_peaks, pre_peak=576, post_peak=624)
                windows_df = ecg_processor.create_dataframe_from_windows_fast(windows)
                predIdxs = ecg_processor.predict(model, windows_df)
            finally:
                os.remove(file_path)  # Geçici dosyayı sil

            # JSON yanıtının oluşturulması
            response = {
                "predIdxs": [int(x) for x in predIdxs.tolist()],
                "index": [int(x) for x in r_peaks_new],
                "signal": [float(x) for x in df_filtered.tolist()]
            }

            return jsonify(response)
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
