import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
import joblib
import librosa
import soundfile as sf
import time
import serial
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import warnings
import tkinter as tk
from tkinter import scrolledtext

# Mengabaikan peringatan spesifik
warnings.filterwarnings("ignore", category=UserWarning, module='librosa')

# Mendefinisikan emosi yang diamati
observed_emotions = ['happy', 'angry']

# Mendefinisikan label emosi untuk pemetaan
emotion_labels = {'happy': 0, 'angry': 1}

# Mendefinisikan emosi dalam dataset RAVDESS
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Mendefinisikan port serial untuk komunikasi dengan Arduino
arduino_port = 'COM3'
arduino_baudrate = 9600
arduino_timeout = 2

# Inisialisasi komunikasi serial dengan Arduino
try:
    arduino = serial.Serial(arduino_port, arduino_baudrate, timeout=arduino_timeout)
    time.sleep(2)  # Allow time for Arduino to initialize
except serial.SerialException as e:
    print(f"Error initializing serial communication: {e}")
    arduino = None

def pad_buffer(buffer, element_size):
    # Menambahkan padding pada buffer untuk memastikan panjangnya merupakan kelipatan dari element_size
    remainder = len(buffer) % element_size
    if remainder > 0:
        padding_size = element_size - remainder
        padding = b'\x00' * padding_size
        buffer += padding
    return buffer

def record_arduino_audio(fs=44100, duration=5):
    if arduino is None:
        print("Arduino not available. Skipping audio recording.")
        return

    print('Recording audio from Arduino...')
    arduino.flushInput()  # Bersihkan data yang ada di buffer serial
    arduino.write(b'r')  # Kirim perintah ke Arduino untuk mulai mengirim data

    frames = []  # Inisialisasi daftar untuk menyimpan frame audio
    start_time = time.time()  # Catat waktu mulai

    while time.time() - start_time < duration:  # Rekam selama durasi yang ditentukan
        # Membaca data dari Arduino
        data = arduino.read(1024)
        frames.append(data)

    arduino.write(b's')  # Kirim perintah ke Arduino untuk berhenti mengirim data
    print('Finished recording audio from Arduino.')

    # Gabungkan daftar objek bytes menjadi satu objek bytes
    frames_bytes = b''.join(frames)

    # Tambahkan padding pada frames_bytes jika perlu untuk memastikan kelipatannya adalah element_size (2 bytes)
    frames_bytes = pad_buffer(frames_bytes, element_size=2)

    # Konversi frame menjadi array numpy
    frames_np = np.frombuffer(frames_bytes, dtype=np.int16)

    # Simpan rekaman audio sebagai file WAV
    sf.write("recorded_audio.wav", frames_np, fs)
    print('Audio saved as recorded_audio.wav.')

def extract_features(file_path, expected_length=205, mfcc=True, chroma=True, mel=True, contrast=True, tonnetz=True, spectral_bandwidth=True, spectral_contrast=True, spectral_flatness=True):
    try:
        with sf.SoundFile(file_path) as sound_file:
            X = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate

            if not np.isfinite(X).all():
                print(f"Audio buffer in {file_path} contains non-finite values.")
                return None
            
            features = []
            if chroma:
                chroma = np.mean(librosa.feature.chroma_stft(y=X, sr=sample_rate, n_fft=1024).T, axis=0)
                features.append(chroma)
            if mfcc:
                mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40, n_fft=1024).T, axis=0)
                features.append(mfccs)
            if mel:
                mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate, n_fft=1024).T, axis=0)
                features.append(mel)
            if contrast:
                contrast = np.mean(librosa.feature.spectral_contrast(y=X, sr=sample_rate, n_fft=1024).T, axis=0)
                features.append(contrast)
            if tonnetz:
                try:
                    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
                    features.append(tonnetz)
                except librosa.util.exceptions.ParameterError:
                    print(f"Skipping tonnetz feature extraction in {file_path} due to non-finite values.")
                    features.append(np.zeros((6,)))  # Append zeros to maintain expected length
            if spectral_bandwidth:
                bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=X, sr=sample_rate, n_fft=1024).T, axis=0)
                features.append(bandwidth)
            if spectral_contrast:
                contrast = np.mean(librosa.feature.spectral_contrast(y=X, sr=sample_rate, n_fft=1024).T, axis=0)
                features.append(contrast)
            if spectral_flatness:
                flatness = np.mean(librosa.feature.spectral_flatness(y=X).T, axis=0)
                features.append(flatness)
            
            # Fitur tambahan
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=X).T, axis=0)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=X, sr=sample_rate).T, axis=0)
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=X, sr=sample_rate).T, axis=0)
            features.extend([zero_crossing_rate, spectral_centroid, spectral_rolloff])

        # Gabungkan dan rata-ratakan fitur
        features = np.concatenate(features)

        # Cek panjang fitur yang diekstrak
        actual_length = len(features)
        if actual_length != expected_length:
            print(f"Feature length mismatch: expected {expected_length}, got {actual_length}. Padding with zeros.")
            if actual_length < expected_length:
                padding = np.zeros(expected_length - actual_length)
                features = np.concatenate([features, padding])
            else:
                features = features[:expected_length]

        return features
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

def load_data(test_size=0.2, augment=True):
    x, y = [], []

    for file in glob.glob("Dataset/Actor_*/*.wav"):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion in observed_emotions:
            feature = extract_features(file, expected_length=205)
            if feature is not None:  # Lewati jika ekstraksi fitur gagal
                x.append(feature)
                y.append(emotion_labels[emotion])
                print(f'Extracted features from {file_name}, emotion: {emotion}')
        else:
            print(f"Ignoring file {file_name}, emotion not in observed_emotions.")

    if not x or not y:
        print("No data loaded. Please check the dataset directory.")
        return None, None, None, None

    print(f'Total samples: {len(y)}')

    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.get_cmap('Blues'))
    plt.show()

def tune_random_forest_hyperparameters(X_train, y_train):
    # Mendefinisikan parameter grid
    param_grid = {
        'randomforestclassifier__n_estimators': [100, 200, 300],
        'randomforestclassifier__max_depth': [None, 10, 20, 30],
        'randomforestclassifier__min_samples_split': [2, 5, 10],
        'randomforestclassifier__min_samples_leaf': [1, 2, 4]
    }

    # Membuat pipeline RandomForestClassifier
    rf_pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=9))

    # Inisialisasi GridSearchCV
    grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Melakukan fit grid search pada data
    grid_search.fit(X_train, y_train)

    # Mengembalikan estimator terbaikr
    return grid_search.best_estimator_

def train_random_forest_model_with_tuning():
    update_output("Training the model...")
    X_train, X_test, y_train, y_test = load_data(test_size=0.2, augment=True)
    if X_train is None or y_train is None or X_test is None or y_test is None:
        update_output("No data to train the model.")
        return

    update_output(f'Training samples: {len(y_train)}, Testing samples: {len(y_test)}')

    # Tune hyperparameters
    best_rf_pipeline = tune_random_forest_hyperparameters(X_train, y_train)

    # Latih model terbaik
    best_rf_pipeline.fit(X_train, y_train)

    # Buat prediksi
    y_pred = best_rf_pipeline.predict(X_test)

    # Evaluasi model
    report = classification_report(y_test, y_pred)
    update_output(report)

    # Hitung akurasi
    accuracy = accuracy_score(y_test, y_pred)
    update_output(f"Accuracy: {accuracy:.2f}")

    # Hitung skor ROC-AUC
    y_prob = best_rf_pipeline.predict_proba(X_test)[:, 1]  # Probability of positive class
    roc_auc = roc_auc_score(y_test, y_prob)
    update_output(f"ROC-AUC Score: {roc_auc:.2f}")

    # Simpan model
    joblib.dump(best_rf_pipeline, 'emotion_model_random_forest_tuned.pkl')
    update_output("Model saved as 'emotion_model_random_forest_tuned.pkl'.")

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, classes=list(emotion_labels.keys()))

    # Plot ROC curve
    plot_roc_curve(y_test, y_prob)

def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def predict_emotion_random_forest():
    try:
        record_arduino_audio()
        features = extract_features("recorded_audio.wav", expected_length=205)

        # Menampilkan pengukuran
        mean_amplitude, mean_dB = calculate_amplitude_and_dB("recorded_audio.wav")
        update_output(f"Mean Amplitude: {mean_amplitude}")
        update_output(f"Mean dB: {mean_dB}")

        if features is None:
            update_output("Feature extraction failed. Unable to predict emotion. Using default features.")
            # Gunakan vektor fitur default jika ekstraksi gagal
            features = np.zeros(205)

        model = joblib.load('emotion_model_random_forest_tuned.pkl')

        # Pastikan array fitur memiliki bentuk yang diharapkan
        expected_feature_length = model.named_steps['standardscaler'].n_features_in_
        actual_feature_length = len(features)
        
        if actual_feature_length != expected_feature_length:
            update_output(f"Feature length mismatch: expected {expected_feature_length}, got {actual_feature_length}")
            # Tambahkan atau potong fitur untuk mencocokkan panjang yang diharapkan
            if actual_feature_length < expected_feature_length:
                padding = np.zeros(expected_feature_length - actual_feature_length)
                features = np.concatenate([features, padding])
            else:
                features = features[:expected_feature_length]

        emotion_label = model.predict([features])[0]

        predicted_emotion = None
        for emotion, label in emotion_labels.items():
            if label == emotion_label:
                predicted_emotion = emotion
                break

        if predicted_emotion is not None:
            update_output(f"Predicted Emotion: {predicted_emotion}")
        else:
            update_output("Unknown emotion predicted.")

        return predicted_emotion
    except Exception as e:
        update_output(f"An error occurred during prediction: {e}")
        return None  

def calculate_amplitude_and_dB(audio_file):
    # Muat file audio
    audio_data, sample_rate = sf.read(audio_file)

    # Hitung rata-rata amplitude
    mean_amplitude = np.mean(np.abs(audio_data))

    # Hitung RMS (Root Mean Square) amplitude untuk setiap frame
    rms_amplitude = np.sqrt(np.mean(audio_data**2, axis=0))

    # Amplitudo referensi untuk perhitungan dB (mengasumsikan audio 16-bit)
    ref_amplitude = 2 ** 15

    # Konversi ke dB
    rms_dB = 20 * np.log10(rms_amplitude / ref_amplitude)

    # Hitung rata-rata level dB
    mean_dB = np.mean(rms_dB)

    return mean_amplitude, mean_dB

output_text = None

def update_output(text):
    global output_text
    if output_text is None:
        print("Error: output_text is not initialized.")
        return
    
    output_text.config(state=tk.NORMAL)
    output_text.insert(tk.END, text + '\n')
    output_text.config(state=tk.DISABLED)
    output_text.see(tk.END)

def clear_output():
    global output_text
    if output_text is None:
        print("Error: output_text is not initialized.")
        return

    output_text.config(state=tk.NORMAL)
    output_text.delete(1.0, tk.END)
    output_text.config(state=tk.DISABLED)

def train_model():
    train_random_forest_model_with_tuning()

def predict_emotion():
    predict_emotion_random_forest()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Emotion Recognition")

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = int(screen_width / 2)
    window_height = int(screen_height / 2)

    root.geometry(f"{window_width}x{window_height}")

    btn_train = tk.Button(root, text="Train Model", command=train_model, width=20, height=2)
    btn_train.pack(pady=10)

    btn_predict = tk.Button(root, text="Predict Emotion", command=predict_emotion, width=20, height=2)
    btn_predict.pack(pady=10)

    btn_clear = tk.Button(root, text="Clear Output", command=clear_output, width=20, height=2)
    btn_clear.pack(pady=10)

    output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED, width=window_width, height=window_height//2)
    output_text.pack(pady=10)

    root.mainloop()
