import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from pathlib import Path

st.set_page_config(page_title="Fish Species Classification",page_icon="🐟",layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path("/content/drive/MyDrive/Fish_Classification_Project/models/efficientnetb0_best.keras")

CLASS_NAMES = [
    "Black Sea Sprat",
    "Gilt-Head Bream",
    "Hourse Mackerel"
]

IMG_SIZE = 256
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH,compile=False)

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

st.title("🐟 Fish Species Classification")
st.write(
    "Upload a fish image to identify its species "
    "using the trained EfficientNetB0 model."
)

with st.sidebar:
    st.header("Model Information")
    st.write("**Model:** EfficientNetB0")
    st.write("**Image size:** 256 × 256")
    st.write("**Number of classes:** 3")
    st.write("**Test accuracy:** 99.56%")

    st.caption(
        "Reported test performance may differ "
        "on new images."
    )

uploaded_file = st.file_uploader("Choose a fish image",type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("Uploaded Image")
    st.image(image,caption="Input fish image",width=400)

    if st.button("Classify Fish", type="primary"):
        if not MODEL_PATH.exists():
            st.error(f"Model file not found: {MODEL_PATH}")
        else:
            try:
                with st.spinner("Predicting fish species..."):
                    model = load_model()
                    input_image = preprocess_image(image)
                    probabilities = model.predict(input_image,verbose=0)[0]

                predicted_index = int(np.argmax(probabilities))
                predicted_species = CLASS_NAMES[predicted_index]
                confidence = float(probabilities[predicted_index])

                st.success(f"Predicted Species: {predicted_species}")
                st.metric("Prediction Confidence",f"{confidence * 100:.2f}%")

                st.subheader("Class Probability Scores")
                for name, probability in zip(CLASS_NAMES,probabilities):
                    st.write(
                        f"**{name}:** "
                        f"{float(probability) * 100:.2f}%"
                    )

                    st.progress(float(np.clip(probability, 0, 1)))

                st.caption(
                    "Confidence is the model's predicted "
                    "probability, not a guarantee that "
                    "the prediction is correct."
                )

            except Exception as error:
                st.error(f"Classification failed: {error}")

else:
    st.info("Upload an image to start classification.")
