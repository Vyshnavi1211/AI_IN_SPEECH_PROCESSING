#!/usr/bin/env python
# coding: utf-8

# In[8]:


import librosa
import librosa.display
import matplotlib.pyplot as plt
import sounddevice as sd

# Load the original audio signal
file_path = '21036-ai.wav'
y, sr = librosa.load(file_path)

# Trim the silence from the beginning and end
y_trimmed, index = librosa.effects.trim(y)

# Create a time array for plotting
time = np.linspace(0, len(y) / sr, len(y))
time_trimmed = np.linspace(0, len(y_trimmed) / sr, len(y_trimmed))

# Plot both the original and trimmed audio signals
plt.figure(figsize=(12, 6))

# Plot the original audio signal
plt.plot(time, y, label='Original Audio Signal', alpha=0.7)

# Plot the trimmed audio signal
plt.plot(time_trimmed, y_trimmed, label='Trimmed Audio Signal', alpha=0.7)

plt.title('Original and Trimmed Audio Signals')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Play original audio
print("Playing original audio...")
sd.play(y, sr)
sd.wait()

# Play trimmed audio
print("Playing trimmed audio...")
sd.play(y_trimmed, sr)
sd.wait()


# In[21]:


## 2) librosa.effects.split()


# In[20]:


import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio

# Load the audio file
file_path = "21036-ai.wav"
signal, sr = librosa.load(file_path, sr=None)

# speech splitting with different top_db values
top_db_values = [19,20, 30, 40]  
split_signals = []

for top_db in top_db_values:
    split_signal = librosa.effects.split(signal, top_db=top_db)
    split_signals.append(split_signal)

# Plot the original and split audio signals for each top_db value
plt.figure(figsize=(12, 8))
plt.subplot(len(top_db_values) + 1, 1, 1)
librosa.display.waveshow(signal, sr=sr)
plt.title('Original Signal')

for i, split_signal in enumerate(split_signals):
    plt.subplot(len(top_db_values) + 1, 1, i + 2)
    split_signal_plot = np.zeros_like(signal)
    for interval in split_signal:
        split_signal_plot[interval[0]:interval[1]] = signal[interval[0]:interval[1]]
    librosa.display.waveshow(split_signal_plot, sr=sr)
    plt.title(f'Split Signal (top_db={top_db_values[i]})')

    # Listen to the split signal
    split_audio = np.concatenate([signal[interval[0]:interval[1]] for interval in split_signal])
    display(Audio(data=split_audio, rate=sr))

plt.tight_layout()
plt.show()


# In[ ]:


## 3) USING IEEE papers


# In[ ]:


Lower top_db values (e.g., 19, 20) result in more aggressive splitting, capturing quieter segments as separate splits. 
This may lead to more fragmented speech with smaller isolated segments.
Moderate top_db values (e.g., 30) offer a balance between granularity and cohesiveness, capturing moderately quiet segments without excessive fragmentation.
Higher top_db values (e.g., 40) lead to less aggressive splitting, capturing only relatively loud segments. 
This may result in fewer, but longer, speech segments.
The choice of top_db depends on the specific characteristics of your audio data and the desired level of granularity in the split signals

