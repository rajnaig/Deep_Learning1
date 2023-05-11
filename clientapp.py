# Import necessary libraries
import streamlit as st
from PIL import Image
import numpy as np

# Load your trained model
from tensorflow.keras.models import load_model



# Define the CIFAR-10 classes
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

model = load_model('models/best_model.h5')

st.title('Deep Learning I.')




uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)  
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    
    # Convert to RGB and preprocess for the model
    image = image.convert('RGB').resize((32, 32))  # Convert to RGB and resize
    image = np.array(image)
    image = image / 255.0  # Your model expects images scaled in the [0, 1] range
    image = np.expand_dims(image, axis=0)  # Add the batch dimension

    # Predict class
    prediction = model.predict(image)

    predicted_class = cifar10_classes[np.argmax(prediction)]



    

    st.markdown(f"# Predicted Class: {predicted_class}")

