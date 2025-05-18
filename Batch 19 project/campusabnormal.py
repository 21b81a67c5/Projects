import os
import cv2
import numpy as np
import pygame
import smtplib
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from email.mime.text import MIMEText

# Dataset Path Configuration
DATASET_PATH = "C:/Users/NAYANA DURGA/Downloads/Avenue_Dataset"  # Update with your dataset path
FRAME_HEIGHT, FRAME_WIDTH = 224, 224
NUM_FRAMES = 32  # Temporal Sequence Length
MODEL_PATH = "tst_anomaly_detector.keras"
THRESHOLD = 0.65  # Adjust as needed

# Initialize Pygame for sound alerts
pygame.mixer.init()

# Buzzer Function
def play_alert_sound():
    """Play buzzer sound for abnormal detection."""
    sound_file = "alert.wav"
    if os.path.exists(sound_file):
        pygame.mixer.music.load(sound_file)
        pygame.mixer.music.play()
    else:
        print("Alert sound file not found!")

# Email Alert Function
def send_email_alert():
    """Send an email alert when abnormal activity is detected."""
    sender_email = "your_email@gmail.com"  # Change to your email
    receiver_email = "recipient_email@gmail.com"  # Change to recipient email
    subject = "🚨 Abnormal Activity Detected!"
    body = "Anomaly detected in the surveillance footage. Please check the video immediately."

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, "your_email_password")  # Update password or use app password
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("Email alert sent successfully.")
    except Exception as e:
        print("Failed to send email:", e)

# Load or Create Model
def create_tst_model():
    base_model = keras.applications.EfficientNetB0(
        input_shape=(FRAME_HEIGHT, FRAME_WIDTH, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False  # Freeze base model layers
    
    model = keras.Sequential([
        layers.TimeDistributed(base_model, input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 3)),
        layers.TimeDistributed(layers.GlobalAveragePooling2D()),
        layers.Bidirectional(layers.LSTM(256, return_sequences=True)),
        layers.GlobalAveragePooling1D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    model = create_tst_model()
    model.save(MODEL_PATH)

# Load Video for Prediction
def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // NUM_FRAMES)

    for i in range(NUM_FRAMES):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frames.append(frame / 255.0)

    cap.release()

    # Pad if not enough frames
    while len(frames) < NUM_FRAMES:
        frames.append(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3)))

    return np.expand_dims(np.array(frames), axis=0)

# Streamlit UI
st.title("Campus Anomaly Detection 🚨")
uploaded_file = st.file_uploader("Upload a surveillance video for analysis", type=["mp4", "avi"])

if uploaded_file is not None:
    file_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.video(file_path)

    if st.button("Analyze Video"):
        with st.spinner("Processing..."):
            try:
                video = load_video(file_path)
                prediction = model.predict(video)[0][0]
                result = "Abnormal" if prediction > THRESHOLD else "Normal"

                st.subheader("Analysis Results:")
                st.write(f"Prediction: {result} (Confidence: {prediction:.2%})")

                if result == "Abnormal":
                    play_alert_sound()
                    send_email_alert()
                    st.error("🚨 Abnormal Activity Detected!")
                else:
                    st.success("✅ Normal Activity")

            except Exception as e:
                st.error(f"Error: {e}")

        os.remove(file_path)

# Debugging Info
st.write("Debug: Model loaded successfully.")