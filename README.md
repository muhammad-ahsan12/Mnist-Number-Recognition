# MNIST Number Recognition Model using CNN

I am Excited to share my recent project . I have created a powerful model that can recognize handwritten numbers with impressive accuracy. Using a special type of neural network called CNN, I trained the model on the MNIST dataset. By carefully adjusting different layers and working with the dataset, I achieved great results.Additionally, I have extended the functionality of this model by integrating it into a user-friendly application using Streamlit. This allows users to upload their own handwritten digit images and receive instant predictions from the trained model

# How to Use

This repository contains code for training a deep learning model using Convolutional Neural Networks (CNN) to recognize handwritten digits from the MNIST dataset. Additionally, it includes a Streamlit application for predicting digits from user-uploaded images.

Requirements
   Python 3.x
   TensorFlow 2.x
   Streamlit
Training the Model
 #Clone this repository to your local machine:
 --->git clone https://github.com/your_username/mnist-cnn-model.git
#Navigate to the project directory:

 --->cd mnist-cnn-model
#Install the required dependencies:
 --->pip install -r requirements.txt
 
#Run the training script to train the CNN model:
 --->python train_model.py
 
This will train the model using the MNIST dataset and save the trained model as model.h5.

Running the Streamlit App 
Once the model is trained, you can run the Streamlit app to predict digits from user-uploaded images.
#Navigate to the project directory and run the Streamlit app:
  --->streamlit run app.py
  
This will start the Streamlit server, and you can access the app in your web browser at http://localhost:8501.
Upload an image of a handwritten digit, and the app will predict the digit using the trained CNN model.
Sample Usage
Train the model using train_model.py.
Run the Streamlit app using streamlit run app.py.
Upload an image of a handwritten digit to the app and see the prediction.
Feel free to customize the content as needed for your specific project setup and requirements!
