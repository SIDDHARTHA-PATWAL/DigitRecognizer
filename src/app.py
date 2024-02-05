from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the MNIST dataset and model outside of the route
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
y_train = y_train.reshape(60000, )

model = tf.keras.models.load_model("handwritten.model")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    # Load the image and convert it to grayscale
    img = Image.open(file).convert('L')


    # Convert the PIL image to a NumPy array
    numpydata = np.array(img)
    inverted_image = 255 - numpydata

    # Reshape the array to match the model input shape
    numpydata = inverted_image.reshape(1, 28 * 28)

    # Make predictions using the loaded model
    prediction = model.predict(numpydata)
    prediction_p = tf.nn.softmax(prediction)
    predicted_digit = np.argmax(prediction_p)
    return render_template("result.html",predicted_digit=predicted_digit)


