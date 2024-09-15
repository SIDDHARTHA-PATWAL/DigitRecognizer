from flask import Flask, render_template, request ,  jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy

app = Flask(__name__)

model = tf.keras.models.load_model("my_model.keras")

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

    try:
        # Load the image and convert it to grayscale
        img = Image.open(file).convert('L')
        # img = Image.open(file)
        # img.show()
        print("SIZE ",img.size )

        # # Resize the image to 28x28 pixels
        # img = img.resize((28, 28))
        print(img)

        # # Convert the PIL image to a NumPy array
        numpydata = np.array(img)
        # print(f"Input data: {numpydata}")
        
        numpydata = numpydata.reshape(1, 28 * 28)
        # # Invert the image
        numpydata = 255 - numpydata
        print(numpydata)

        # # Debugging: Print the shape and contents of numpydata
        print(f"Shape of input data: {numpydata.shape}")
        # print(f"Input data: {numpydata}")

        # # Make predictions using the loaded model
        prediction = model.predict(numpydata.reshape(1,784))
        prediction_p = tf.nn.softmax(prediction).numpy()

        # Debugging: Print the raw prediction values
        print(f"Raw prediction: {prediction}")
        print(f"Softmax prediction: {prediction_p}")

        predicted_digit = np.argmax(prediction_p)
        print(f"argmax prediction: {predicted_digit}")
        # return jsonify(numpydata.tolist())
        # predicted_digit = 9
        return render_template("result.html", predicted_digit=predicted_digit, confidence=prediction_p[0][predicted_digit]*100)
    except Exception as e:
        return f"Error processing the file: {str(e)}"

if __name__ == "__main__":
    app.run()
