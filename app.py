
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import io

st.set_page_config(page_title="Fish U-Net Segmentation",page_icon="🐟",layout="wide")

st.title("🐟 Fish U-Net Segmentation")
st.write(
    "Upload a fish image to generate a segmentation mask "
    "using the trained U-Net model."
)

st.divider()

MODEL_PATH = "fish_unet_final.keras"
@st.cache_resource
def load_fish_model():
    model = tf.keras.models.load_model(MODEL_PATH,compile=False)
    return model
model = load_fish_model()
with st.sidebar:
    st.header("Model Information")
    st.write("**Model:** Fish U-Net")
    st.write("**Input Size:** 256 × 256")
    st.write("**Output:** Binary Segmentation")
    st.write("### Test Performance")
    st.metric("Test Dice","0.9692")
    st.metric("Test IoU","0.9448")
uploaded_file = st.file_uploader("Upload a fish image",type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")
    resized_image = original_image.resize((256, 256),Image.Resampling.BILINEAR)
    image_array = np.array(resized_image,dtype=np.float32)
    image_array = image_array / 255.0
    input_tensor = np.expand_dims(image_array,axis=0)
    with st.spinner("Segmenting fish..."):
        prediction = model.predict(input_tensor,verbose=0)
    predicted_mask = (prediction[0, :, :, 0] >= 0.5).astype(np.uint8)
    st.subheader("Segmentation Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(original_image,caption="Original Fish Image",use_container_width=True)
    with col2:
      mask_image = Image.fromarray((predicted_mask * 255).astype(np.uint8))
      import io
      buffer = io.BytesIO()
      mask_image.save(buffer,format="PNG")
      st.download_button(label="⬇️ Download Segmentation Mask",data=buffer.getvalue(),file_name="fish_segmentation_mask.png",mime="image/png")
      st.image(mask_image,caption="Predicted Segmentation",use_container_width=True)
    with col3:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(image_array)
        ax.imshow(predicted_mask,cmap="jet",alpha=0.4)
        ax.axis("off")
        ax.set_title("Segmentation Overlay")
        st.pyplot(fig,use_container_width=True)
        overlay_buffer = io.BytesIO()
        fig.savefig(overlay_buffer,format="PNG",bbox_inches="tight",pad_inches=0)
        overlay_buffer.seek(0)
        plt.close(fig)
        st.download_button(label="⬇️ Download Overlay",data=overlay_buffer.getvalue(),file_name="fish_segmentation_overlay.png",mime="image/png")

    st.divider()
    st.subheader("Prediction Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Input Size:** 256 × 256")
    with col2:
        st.write("**Threshold:** 0.5")
    with col3:
        st.write("**Output:** Binary Mask")
else:
    st.info("Please upload a fish image to start segmentation.")
