# whisper_client.py
import socket
import os
import whisper
import numpy as np
import sounddevice as sd
import sys

# Configuration
ROBOT_IP = os.environ.get('ROBOT_IP', '127.0.0.1')  # Replace with your Jetson's IP
ROBOT_PORT = 5000
SAMPLE_RATE = 16000
RECORD_SECONDS = 10  # Adjust based on your command length

# Load Whisper model
print("Loading Whisper model...")
model = whisper.load_model("base")  # Options: tiny, base, small, medium, large
print("Model loaded!")

def record_audio(duration=RECORD_SECONDS):
    """Record audio from microphone"""
    print(f"🎤 Recording for {duration} seconds...")
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.float32
    )
    sd.wait()
    print("Recording complete.")
    return audio.flatten()

def transcribe(audio):
    """Transcribe audio using Whisper"""
    result = model.transcribe(
        audio,
        language="en",  # Change if needed, or remove for auto-detect
        fp16=False
    )
    return result["text"].strip()

def send_to_robot(text, sock):
    """Send text command to robot"""
    try:
        sock.sendall(text.encode('utf-8'))
        response = sock.recv(1024)
        print(f"Robot: {response.decode()}")
    except Exception as e:
        print(f"Send error: {e}")

def main():
    # Connect to robot
    print(f"Connecting to robot at {ROBOT_IP}:{ROBOT_PORT}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        sock.connect((ROBOT_IP, ROBOT_PORT))
        print("Connected to robot!\n")
    except ConnectionRefusedError:
        print("Could not connect to robot. Is the server running?")
        sys.exit(1)
    
    print("Press Enter to record a command, or 'q' to quit.\n")
    
    try:
        while True:
            user_input = input("Press Enter to speak... ")
            if user_input.lower() == 'q':
                break
            
            # Record and transcribe
            audio = record_audio()
            text = transcribe(audio)
            
            if text:
                print(f"You said: \"{text}\"")
                send_to_robot(text, sock)
            else:
                print("No speech detected.")
            print()
    
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        sock.close()

if __name__ == "__main__":
    main()