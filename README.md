# Emotion_detection
Facial Emotion Recognition
This repository contains the code for a Facial Emotion Recognition (FER) system that can detect human emotions from images. The project includes a deep learning model trained on facial images and a simple Graphical User Interface (GUI) to allow users to upload images and see the predicted emotion.

Table of Contents
Features

Emotion Categories

Model Architecture

Dataset

Installation

Usage

Training the Model

Using the GUI Application

Project Structure

Results

Contributing

License

Features
Emotion Detection: Identifies 7 universal human emotions from facial images.

Deep Learning Model: Utilizes a Convolutional Neural Network (CNN) for robust emotion classification.

GUI Application: A user-friendly interface built with tkinter for easy image upload and real-time emotion prediction.

Pre-trained Weights: Includes functionality to load a pre-trained model for quick setup and inference.

Emotion Categories
The model is trained to classify the following 7 emotions:

Angry

Disgust

Fear

Happy

Neutral

Sad

Surprise

Model Architecture
The deep learning model is a Convolutional Neural Network (CNN) implemented using TensorFlow/Keras. It consists of multiple convolutional blocks, each comprising:

Conv2D: Convolutional layer for feature extraction.

BatchNormalization: To stabilize and accelerate training.

Activation('relu'): Rectified Linear Unit activation function.

MaxPooling2D: For down-sampling and reducing dimensionality.

Dropout: To prevent overfitting.

These blocks are followed by a Flatten layer and dense (fully connected) layers with a softmax activation in the final layer for multi-class classification. The model is compiled with Adam optimizer and categorical_crossentropy loss.

Dataset
The model is trained on a dataset of facial images categorized into the 7 emotions mentioned above. The dataset is expected to be organized in the following directory structure:

Emotion_detection/
├── train/
│   ├── angry/
│   ├── disgusted/
│   ├── fearful/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgusted/
    ├── fearful/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/

Images for each emotion should be placed in their respective subdirectories within train/ and test/ (or validation/).

Installation
Clone the repository:

git clone https://github.com/your-username/Emotion_Detection.git
cd Emotion_Detection

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`

Install the required libraries:

pip install tensorflow matplotlib opencv-python pillow scikit-learn

Note: Ensure you have opencv-python installed, as cv2 is used for image processing.

Download haarcascade_frontalface_default.xml:
The GUI application uses OpenCV's Haar Cascade classifier for face detection. Download the haarcascade_frontalface_default.xml file and place it in the project's root directory. You can find it here.

Place your dataset:
Organize your emotion dataset into train/ and test/ (or validation/) directories as described in the Dataset section, within the project root.

Usage
Training the Model
The model_creation.ipynb Jupyter Notebook guides you through the process of:

Loading and exploring the dataset.

Defining the CNN model architecture.

Preparing the data using ImageDataGenerator.

Training the model.

Evaluating the model's performance.

Saving the model's architecture (model_a.json) and weights (model_weights.h5).

To run the notebook:

jupyter notebook model_creation.ipynb

Using the GUI Application
The gui.py script provides a simple Tkinter-based GUI for real-time emotion detection on uploaded images.

Ensure you have model_a.json (model architecture) and model_weights.h5 (trained weights) in the same directory as gui.py. These files are generated after training the model using model_creation.ipynb.

Ensure haarcascade_frontalface_default.xml is in the project root.

Run the GUI application:

python gui.py

The application window will open. Click "Upload Image" to select a facial image, and the detected emotion will be displayed.

Project Structure
.
├── model_creation.ipynb    # Jupyter notebook for model training and evaluation
├── gui.py                  # Tkinter-based GUI application for inference
├── haarcascade_frontalface_default.xml # Haar Cascade file for face detection
├── model_a.json            # Saved Keras model architecture (JSON format)
├── model_weights.h5        # Saved Keras model weights (H5 format)
├── train/                  # Training dataset directory
│   ├── angry/
│   └── ... (other emotion folders)
└── test/                   # Testing/Validation dataset directory
    ├── angry/
    └── ... (other emotion folders)

Results
After 15 epochs of training, the model achieved approximately:

Training Accuracy: ~72.02%

Validation Accuracy: ~59.93%

(Note: These results are based on the output logs in model_creation.ipynb. Actual results may vary slightly based on hardware and TensorFlow version.)

Plots for model accuracy and loss during training are generated in the notebook.

Contributing
Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please open an issue or submit a pull request.

License
This project is open-source and available under the MIT License.
