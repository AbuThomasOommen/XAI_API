from flask import Flask, render_template, request, jsonify, send_file
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageChops, ImageEnhance
from skimage import transform
import matplotlib.pyplot as plt
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from lime.lime_text import LimeTextExplainer
from skimage.segmentation import mark_boundaries
from model_codes.preprocessing_text import TextsToSequences,Padder,create_model
import io
import joblib

app = Flask(__name__)

# Load the saved model
cnn_model = tf.keras.models.load_model('models/cnn_model.h5')
nlp_pipeline = joblib.load('models/pipeline_model.joblib')

def process_img(img):
    image = cv2.imdecode(np.frombuffer(img.read(), np.uint8), cv2.IMREAD_COLOR)
    image = np.array(image).astype('float32')/255
    image = transform.resize(image,(128,128,3))
    image = np.expand_dims(image,axis=0)    
    return image


@app.route("/", methods=['GET','POST'])
@app.route("/home", methods=['GET','POST'])
def index():
    print("In the index")
    return render_template("index5.html")

@app.route('/predict', methods=['POST'])
def upload():
    print("In the predict route")
    if request.method == 'POST' and 'imageInput' in request.files:
        image = request.files['imageInput']
        print("Received for predict image:", image.filename)
        sample = process_img(image)
        print(cnn_model.predict(sample))

        if cnn_model.predict(sample) > .5 :
            msg = "Real Image detected"
            print(msg)
        else:
            msg = "Fake Image detected"
            print(msg)
        
        # Create a JSON object with the message
        response = {'message': msg}

    elif request.method == 'POST' and 'tweetInput' in request.form:
        print("Inside the else")
        class_names = ['Neutral', 'Positive', 'Negative']
        tweet = request.form['tweetInput']
        # tweet_text = tweet.read().decode('utf-8')
        print("Received for classify tweet:", tweet)

        # Make predictions using the loaded pipeline
        predictions = nlp_pipeline.predict([tweet])
        print(predictions)

        # Create a JSON object with the message
        # response = {'message': predictions.tolist()}
        response = {'message': class_names[predictions[0]]}
        
    return jsonify(response)

@app.route('/explain', methods=['POST'])
def explain():
    print("In the explain route")
    if request.method == 'POST' and 'imageInput' in request.files:
        image = request.files['imageInput']
        print("Received for explain image:", image.filename)
        sample = process_img(image)
        sample = np.squeeze(sample)

        # Lime explanations
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(sample, cnn_model.predict, top_labels=2)

        label = 0

        # Visualize the explanations
        temp, mask = explanation.get_image_and_mask(label, positive_only=False, num_features=20, hide_rest=False)

        # Custom color mapping
        positive_color = (0.0, 1.0, 0.0)  # Red color for positive contributions
        negative_color = (1.0, 0.0, 0.0)  # Green color for negative contributions

        # Apply custom colors to the masked image
        masked_image = mark_boundaries(temp, mask, outline_color=positive_color, mode='thick')
        masked_image[mask == 0] = negative_color  # Change the color for negative contributions
        masked_image = transform.resize(masked_image,(256,256,3))
    
        # Convert the image to bytes
        img_bytes = io.BytesIO()
        plt.imsave(img_bytes, masked_image, format='jpeg')
        img_bytes.seek(0)

        # Return the image as a response
        return send_file(img_bytes, mimetype='image/jpeg')
    
    elif request.method == 'POST' and 'tweetInput' in request.form:
        class_names = ['neutral', 'positive', 'negative']
        tweet = request.form['tweetInput']
        # tweet_text = tweet.read().decode('utf-8')
        explainer = LimeTextExplainer(class_names=class_names)
        exp = explainer.explain_instance(tweet, nlp_pipeline.predict_proba, num_features=6, top_labels=3)

        # Convert the image to bytes
        img_bytes = io.BytesIO()
        exp.as_pyplot_figure().savefig(img_bytes, format='jpeg')
        img_bytes.seek(0)

        # Return the image as a response
        return send_file(img_bytes, mimetype='image/jpeg')


if __name__ == "__main__":
    app.run()
