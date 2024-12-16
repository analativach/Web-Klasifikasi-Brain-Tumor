import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import tensorflow as tf
from PIL import Image
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Set the upload folder for temporary image storage
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model
MODEL_PATH = 'training_efficiennet.h5'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Mapping kelas sesuai dengan model pelatihan
CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "Sehat"]

@app.route('/')
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
    """Process the uploaded image and make predictions."""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        # Process the image and make predictions
        prediction = process_image(filepath, model)
        return render_template('predict.html', hasil=prediction, img_url=url_for('uploaded_file', filename=filename))
    except Exception as e:
        print(f"Error processing image: {e}")
        return "Error in prediction process. Please try again."

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def process_image(filepath, model):
    """
    Processes the uploaded image and makes a prediction using the trained model.

    Args:
        filepath: Path to the uploaded image file.
        model: The trained model.

    Returns:
        A string indicating the prediction result.
    """
    try:
        # Load and preprocess the image
        img = Image.open(filepath).convert('RGB').resize((224, 224))  # Ensure RGB and resize
        img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
        
        # Ensure the array has the correct dimensions (1, 224, 224, 3)
        if img_array.ndim == 3:
            img_array = np.expand_dims(img_array, axis=0)

        # Make predictions
        predictions = model.predict(img_array)  # Output probabilities
        predicted_class = np.argmax(predictions, axis=1)[0]  # Get the class index
        predicted_label = CLASS_NAMES[predicted_class]  # Map index to class name
        confidence = predictions[0][predicted_class]  # Confidence of the predicted class

        # Format the result
        result = f"Hasil Prediksi: {predicted_label} (Confidence: {confidence:.2f})"
        return result
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise e

if __name__ == '__main__':
    app.run(debug=True)