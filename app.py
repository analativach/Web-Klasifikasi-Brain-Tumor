import os
import requests
import json
import sys
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import numpy as np
from werkzeug.urls import unquote
from pyngrok import ngrok
from dotenv import load_dotenv


load_dotenv()

def create_app():
    app = Flask(__name__)

    app.config.from_mapping(
        BASE_URL="http://localhost:5000",
        USE_NGROK=os.environ.get("USE_NGROK", "False") == "True"
    )

    if app.config["USE_NGROK"] and os.environ.get("NGROK_AUTHTOKEN"):

        port = sys.argv[sys.argv.index(
            "--port") + 1] if "--port" in sys.argv else "5000"

        public_url = ngrok.connect(port).public_url
        print(
            f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{port}\"")

        app.config["BASE_URL"] = public_url
        init_webhooks(public_url)

    return app


def init_webhooks(base_url):
    url = "https://api.short.io/links/lnk_4QHP_96cVJ1WHYuZjUHSDU1VXz"

    payload = json.dumps({"allowDuplicates": False,
                          "domain": "share.aliepr.my.id",
                          "path": 'braintumorclassifier',
                          'originalURL': base_url,
                          })
    headers = {
        'accept': "application/json",
        'content-type': "application/json",
        'authorization': os.environ.get("SHORTIO_API_KEY")
    }
    response = requests.post(url, data=payload, headers=headers)
    print(response.text)


app = create_app()

# Set the upload folder for temporary image storage
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Mapping kelas sesuai dengan model pelatihan
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Load the pre-trained model
model = load_model('efficientnetb0_tumor_model.h5')


def predict_label(img_path):
    """Predict the class of an image."""
    # Load and preprocess the image
    # Load gambar dengan ukuran sesuai input model
    img = image.load_img(img_path, target_size=(224, 224))

    # Konversi gambar menjadi array
    img_array = image.img_to_array(img)

    # Normalisasi input (gunakan preprocess_input untuk preprocessing model bawaan EfficientNet)
    img_array = preprocess_input(img_array)

    # Tambahkan dimensi batch (dimensi pertama)
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi menggunakan model
    predictions = model.predict(img_array)[0]

    # Cari indeks kelas dengan probabilitas tertinggi
    class_idx = np.argmax(predictions)
    class_name = CLASS_NAMES[class_idx]

    # Dapatkan probabilitas prediksi
    class_prob = predictions[class_idx]

    # Dictionary penjelasan diagnosis
    diagnosis_explanations = {
        'glioma': 'Glioma adalah jenis tumor otak yang berasal dari sel glial. Penanganan tergantung pada ukuran dan lokasi tumor.',
        'meningioma': 'Meningioma adalah tumor otak yang umumnya bersifat jinak, berasal dari jaringan di sekitar otak dan sumsum tulang belakang.',
        'notumor': 'Tidak terdeteksi adanya tumor otak. Namun, selalu ingat untuk tetap menjaga kesehatan tubuh dengan pola hidup sehat. Konsultasikan dengan dokter untuk pemeriksaan lanjutan jika Anda memiliki gejala. Kesehatan adalah aset terbesar, jadi jangan ragu untuk melakukan pengecekan rutin dan menjaga keseimbangan hidup yang baik!',
        'pituitary': 'Pituitary tumor adalah tumor kelenjar hipofisis yang dapat mempengaruhi fungsi hormonal tubuh.'
    }

    # Penjelasan berdasarkan prediksi
    explanation = diagnosis_explanations.get(
        class_name, 'Penjelasan tidak tersedia untuk kelas ini.')

    # Output nama kelas dan probabilitas
    return (
        f"Anda Menderita: {class_name} <br> (Probabilitas Diagnosa : {predictions[class_idx] * 100:.2f}%) <br> Penjelasan: {
            explanation} <br> <span class='warning-text'>Peringatan! AI ini bisa saja salah, jangan jadikan hasilnya sebagai diagnosa utama!</span>"
    )


@app.route('/', methods=['GET', 'POST'])
def index():
    """Render the home page."""
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Handle file uploads."""
    if request.method == 'POST':
        if 'img' not in request.files:
            return "No file part in the request.", 400

        file = request.files['img']
        if file.filename == '':
            return "No file selected.", 400

        if file:
            # Save the uploaded file
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"File uploaded successfully: {filepath}")
            return redirect(url_for('predict', filename=filename))
    return render_template('upload.html')


@app.route('/predict/<filename>', methods=['GET'])
def predict(filename):
    """Predict and render results."""
    # Ambil path gambar
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Lakukan prediksi
    try:
        hasil = predict_label(filepath)
    except Exception as e:
        hasil = f"Error during prediction: {e}"

    return render_template('predict.html', hasil=hasil, data=url_for('uploaded_file', filename=filename))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
