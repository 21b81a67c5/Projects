import numpy as np
import scipy.io.wavfile as wav
import pygame

# Sound parameters
sample_rate = 44100  # Samples per second
duration = 2  # Seconds
frequency = 1000  # Hz (Pitch of the beep)

# Generate waveform (sinusoidal wave)
t = np.linspace(0, duration, int(sample_rate * duration), False)
waveform = (32767 * np.sin(2 * np.pi * frequency * t)).astype(np.int16)

# Save the sound as a WAV file
sound_file = "alert.wav"
wav.write(sound_file, sample_rate, waveform)

# Initialize pygame mixer and play the sound
pygame.mixer.init()
alert_sound = pygame.mixer.Sound(sound_file)
alert_sound.play()

print(f"✅ Alert sound generated and saved as {sound_file}")