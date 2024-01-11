from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import tempfile
import keras
import uuid
from services import ECGProcessor
from flask_cors import CORS


ecg_processor = ECGProcessor()

# Modelin yüklenmesi ve hazırlanması
model_path = "model_04012024_original_label_synthetic.h5"

model = keras.models.load_model(model_path, compile=False)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
value_to_letter = {0: "A", 1: "Q", 2: "V", 3: "Z"}
temp_directory = tempfile.gettempdir()

app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file_record' not in request.files:
            return "No file_record part", 400
        if 'file_sample' not in request.files:
            return "No file_sample part", 400
        file_record = request.files['file_record']
        file_sample = request.files['file_sample']
        frequency = int(request.values['frequency'])
        if file_record.filename == '':
            return "No selected file_record", 400
        if file_sample.filename == '':
            return "No selected file_sample", 400
        if frequency == '':
            return "No frequency", 400
        if file_record:
            record_filename = secure_filename(str(uuid.uuid4()) + file_record.filename)
            record_file_path = os.path.join(temp_directory, record_filename)
            file_record.save(record_file_path)
            print(record_file_path)
        if file_sample:
            sample_filename = secure_filename(str(uuid.uuid4()) + file_sample.filename)
            sample_file_path = os.path.join(temp_directory, sample_filename)
            file_sample.save(sample_file_path)
            print(sample_file_path)
            # Dosya işleme
            try:
                # Dosya uzantısına göre işlem yapma
                if record_file_path.endswith('.txt'):
                    df = ecg_processor.read_txt_file_and_extract_first_column(record_file_path)
                    df = ecg_processor.signal_resample(df, frequency)
                elif record_file_path.endswith('.csv'):
                    df = ecg_processor.read_csv_file_and_extract_first_column(record_file_path)
                    df = ecg_processor.signal_resample(df, frequency)
                    rpeaks = ecg_processor.read_csv_file_and_extract_first_samples(sample_file_path)
                else:
                    raise ValueError("Unsupported file type")

                r_peaks, df_filtered, df_filtered_draw = ecg_processor.process_ecg_data(df)

                windows, r_peaks_new = ecg_processor.window_ecg_signal(df_filtered, rpeaks, pre_peak=144, post_peak=216)
                windows_df = ecg_processor.create_dataframe_from_windows_fast(windows)
                windows_df_normalized = ecg_processor.normalize_rows(windows_df)
                predIdxs = ecg_processor.predict(model, windows_df)
                modified_predIdxs = [value_to_letter[value] for value in predIdxs]

            finally:
                os.remove(record_file_path)  # Geçici dosyayı sil
                os.remove(sample_file_path)  # Geçici dosyayı sil


            # JSON yanıtının oluşturulması
            response = {
                "predIdxs": modified_predIdxs,
                "index": [int(x) for x in r_peaks_new],
                "signal": [float(x) for x in df_filtered_draw.tolist()]
            }

            return jsonify(response)
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
