from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Load the pre-trained EfficientNet model
MODEL_PATH = 'training_efficiennet.h5' 
model = load_model(MODEL_PATH)

# Kategori hasil prediksi
categories = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Folder untuk upload gambar
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Check if file is uploaded
        if 'file' not in request.files:
            return "No file part", 400

        file = request.files['file']

        if file.filename == '':
            return "No selected file", 400

        if file:
            # Save file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # Preprocess image
                img = tf.keras.utils.load_img(filepath, target_size=(224, 224))  # Sesuaikan dengan input model Anda
                img_array = tf.keras.utils.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0  # Normalisasi jika diperlukan

                # Predict
                prediction = model.predict(img_array)
                predicted_class = np.argmax(prediction, axis=1)

                # Cek validitas index hasil prediksi
                if len(predicted_class) > 0 and predicted_class[0] < len(categories):
                    result = categories[predicted_class[0]]
                else:
                    return "Kategori tidak diketahui atau model error.", 500

                # Menentukan hasil berdasarkan kategori
                if result == 'Glioma':
                    return f"Penyakit: {result}"
                elif result == 'No Tumor':
                    return "Anda tidak memiliki penyakit."
                elif result == 'Meningioma':
                    return f"Penyakit: {result}"
                elif result == 'Pituitary':
                    return f"Penyakit: {result}"
                else:
                    return "Kategori tidak diketahui."
            except Exception as e:
                return f"Terjadi error saat memproses gambar: {str(e)}", 500

    return '''
        <!doctype html>
        <title>Prediksi Penyakit</title>
        <h1>Unggah Gambar MRI</h1>
        <form method="POST" enctype="multipart/form-data">
          <input type="file" name="file">
          <input type="submit" value="Unggah dan Prediksi">
        </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
