import streamlit as st
import speech_recognition as sr
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# Load the CSV file containing emotions data
csv_file = 'emotions.csv'  # Update path if needed
data = pd.read_csv(csv_file)

# Split the data into features and labels
X = data['content']
y = data['sentiment']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a text classification pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Function to predict emotion from text
def predict_emotion(text):
    prediction = model.predict([text])
    return prediction[0]

# Streamlit app starts here
st.set_page_config(page_title="Speech to Emotion Detection", layout="wide")

# Add the image at the top of the page with reduced size
st.image("images.jpeg", use_column_width=False, width=400)  # Adjust the width as needed

# Add a header for the app
st.title("🎙️ Speech to Emotion Detection System")

# Columns for speech input and text input
col1, col2 = st.columns(2)

# Sidebar for storing results (Emotion)
st.sidebar.header("Predicted Emotion")
predicted_emotion = st.sidebar.text_area("Emotion:", "Predicted emotion will appear here...")

# Speech recognition and emotion prediction function
def recognize_speech():
    r = sr.Recognizer()

    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        # Adjust for ambient noise for better accuracy
        r.adjust_for_ambient_noise(source)
        st.write("Please speak now...")
        audio = r.listen(source, timeout=5, phrase_time_limit=15)

        try:
            # Use Google Web Speech API to recognize speech
            text = r.recognize_google(audio)
            st.success(f"Transcription: {text}")
            emotion = predict_emotion(text)
            st.sidebar.text_area("Emotion:", emotion)

        except sr.UnknownValueError:
            st.error("Sorry, I couldn't understand what you said.")
        
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")

# Column 1: Speech Input
with col1:
    st.subheader("🎤 Speak")
    if st.button("Start Speaking"):
        recognize_speech()

# Column 2: Text Input
with col2:
    st.subheader("📝 Text Input")
    input_text = st.text_area("Enter your text here...")
    
    if st.button("Analyze Text"):
        if input_text:
            emotion = predict_emotion(input_text)
            st.sidebar.text_area("Emotion:", emotion)
        else:
            st.error("Please enter some text for analysis.")
