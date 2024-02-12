#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa
import librosa.display


# # A1.Load the recorded speech file into your python workspace. Once loaded, plot the graph for the speech signal.You may use the below code from librosa asa reference.

# In[14]:


filename='21036-ai.wav'
y, sr = librosa.load(filename)
librosa.display.waveshow(y)
print(sr)


# In[16]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from glob import glob
import librosa
import librosa.display
import IPython.display as ipd
from itertools import cycle

sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
# Trimming leading/lagging silence
y_trimmed, _ = librosa.effects.trim(y, top_db=20)
pd.Series(y_trimmed).plot(figsize=(10, 5),
                  lw=1,
                  title='Raw Audio Trimmed Example',
                  color=color_pal[4])
plt.show()


# # A3. Take a small segment of the signal and play it. 

# In[19]:


start_time = 0.5
end_time = 2.0
start_sample = int(start_time * sr)
end_sample = int(end_time * sr)
y_segmented = y[start_sample:end_sample]
plt.figure(figsize=(10, 5))
plt.plot(y_segmented, color='purple', label='Segmented Audio')
plt.title('Segmented Audio Example')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()
plt.show()


# # A4. Play around with your recorded speech signal for various segments. Understand the nature of the signal. Also observe with abruptly segmented speech, how the perception of that speech is affected.

# In[20]:


import librosa
import sounddevice as sd
import time
filename = '21036-ai.wav'
y, sr = librosa.load(filename)
starting_time = 0.5
ending_time = 2.0
segment = y[int(starting_time * sr):int(ending_time * sr)]
ipd.Audio(segment, rate=sr)

