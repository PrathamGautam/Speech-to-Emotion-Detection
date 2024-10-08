# 🎤 Speech-to-Emotion Detection System 🎭

This is a **Speech-to-Emotion Detection System** built using **Streamlit**, **Speech Recognition**, and **Machine Learning**. The system converts speech or text input into emotions by predicting the emotional tone of the input. This project leverages a trained Naive Bayes classifier to detect emotions like happiness, sadness, anger, etc.

## 🚀 Features

- **🎙 Speech Recognition**: The system listens to the user's speech using a microphone and transcribes it into text.
- **📝 Text Emotion Analysis**: Users can input text manually, and the system will predict the emotion associated with the text.
- **🤖 Emotion Prediction**: The emotion prediction is made using a machine learning model trained on emotion-labeled text data.
- **🌐 User Interface**: Built with **Streamlit**, the web app is simple and interactive. It displays the emotion prediction in the sidebar.

## 🔧 Prerequisites

- **Python 3.x** or higher
- Libraries:
  - `streamlit`
  - `speechrecognition`
  - `pandas`
  - `scikit-learn`
  - `numpy`

To install these dependencies, use the following command:

```bash
pip install streamlit speechrecognition scikit-learn pandas numpy
```
## 🖥 Application Layout
- 🖼 Top Image: A visually appealing image of sound waves, placed at the top.
- 🎤 Left Section: This section allows users to speak and have the system transcribe their speech and predict the emotion.
- 📝 Right Section: This section allows users to enter text for emotion analysis.
- 📊 Sidebar: Displays the predicted emotion.

## 🛠 Technologies Used
- 🎙 Speech Recognition: Used to convert speech to text using the Google Web Speech API.
- 🌐 Streamlit: For creating the interactive web app.
- 🤖 Naive Bayes Classifier: The machine learning model used to predict emotions from the text input.
- 📊 Pandas: For data manipulation and handling the CSV dataset.
- 🧠 Scikit-learn: For building and training the machine learning model.
