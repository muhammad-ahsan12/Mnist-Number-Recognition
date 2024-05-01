import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('new_model.h5')  # Replace 'your_model.h5' with the path to your trained model

# Define class names for the MNIST dataset
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Add Font Awesome icons and custom button style (optional)
st.markdown(
    """
    <style>
        .fas {
            font-family: 'Font Awesome 5 Free';
            font-weight: 900;
        }
        div.stButton>button {
            background-color: #4CAB50;
            color: white;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            transition-duration: 0.4s;
            cursor: pointer;
            border-radius: 8px;
            border: none;
        }
        div.stButton>button:hover {
            background-color: #45a049;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
def main():
    st.title("🤖MNIST Digit Recognition")

    # Optional: Short description of the app
    st.write(
        """
        This app helps you predict the digit in a handwritten image using a trained deep learning model.
        """
    )

    # Upload image in sidebar
    uploaded_file = st.sidebar.file_uploader("Upload an image of a handwritten digit", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display error message if no image is uploaded and user clicks predict
        if st.sidebar.button("Predict"):
            if uploaded_file is None:
                st.error("No image uploaded. Please upload an image of a handwritten digit.")
            else:
                # Preprocess the image for prediction
                image = Image.open(uploaded_file)
                img_array = np.array(image.resize((28, 28)).convert('L'))  # Resize and convert to grayscale
                img_array = img_array / 255.0  # Normalize pixel values
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
                img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension

                # Make prediction
                prediction = model.predict(img_array)
                predicted_label = np.argmax(prediction)

                # Display uploaded image and predicted label in main area
                st.image(image, caption="Uploaded Image", width=300)
                st.success(f"The Number is: {class_names[predicted_label]}")

if __name__ == "__main__":
    main()
