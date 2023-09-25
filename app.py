import base64
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
from classnames import class_names

app = Flask(__name__)

# Load the TensorFlow model
model = tf.keras.models.load_model("model.h5")

input_shape = (224, 224)


def image_preprocessing(image):
    img = Image.open(image)
    # Resize the image to the target input shape (224x224)
    img = img.resize(input_shape)
    img_array = np.array(img)
    return img_array


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    img_src = None
    confidence = None
    if request.method == "POST":
        file = request.files["file"]
        processed_image = image_preprocessing(file)
        prediction = model.predict(np.expand_dims(processed_image, axis=0))
        class_label = np.argmax(prediction)

        processed_image_base64 = base64.b64encode(processed_image).decode("utf-8")

        result = f"Prediction: {class_names[class_label]}"
        confidence = round(prediction[0][class_label] * 100, 2)
        img_src = f"data:image/jpeg;base64,{processed_image_base64}"

    return render_template(
        "index.html", result=result, img_src=img_src, confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)
