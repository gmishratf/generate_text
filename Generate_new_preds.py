
# coding: utf-8

# In[1]:


import sys
import numpy
import re
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


# In[2]:


f1 = "./Data/001ssb.txt"
f2 = "./Data/002ssb.txt"
f3 = "./Data/003ssb.txt"
f4 = "./Data/004ssb.txt"
f5 = "./Data/005ssb.txt"


# In[3]:


raw_text = open(f1).read() + open(f2).read() + open(f3).read() + open(f4).read() + open(f5).read()


# In[4]:


raw_text = raw_text.lower()


# In[5]:


raw_text.replace(r'Page (\d+)\n', '')


# In[7]:


chars = sorted(list(set(raw_text)))
char_to_int = dict((c,i) for i,c in enumerate(chars))


# In[8]:


n_chars = len(raw_text)
n_vocab = len(chars)


# In[9]:


seq_length = 100
dataX = []
dataY = []


# In[10]:


for i in tqdm(range(0, n_chars - seq_length, 1), desc="Building datasets"):
    seq_in = raw_text[i:i+seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])


# In[13]:


n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)


# In[14]:


X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
X = X/float(n_vocab)
y = np_utils.to_categorical(dataY)


# In[15]:


model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))


# In[16]:


filepath = "./Checkpoints/weights-improvement-04-1.46159-bigger.hdf5"


# In[17]:


model.load_weights(filepath)


# In[18]:


model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[19]:


int_to_char = dict((i, c) for i, c in enumerate(chars))


# In[20]:


start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")


# In[ ]:


for i in range(1000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]

