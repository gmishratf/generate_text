
# coding: utf-8

# In[ ]:


import re
import numpy
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


# In[ ]:


f1 = "./Data/001ssb.txt"
f2 = "./Data/002ssb.txt"
f3 = "./Data/003ssb.txt"
f4 = "./Data/004ssb.txt"
f5 = "./Data/005ssb.txt"


# In[ ]:


EMBEDDING_FILE = "./Data/glove.840B.300d.txt"


# In[ ]:


raw_text = open(f1).read() + open(f2).read() + open(f3).read() + open(f4).read() + open(f5).read()


# In[ ]:


raw_text = raw_text.lower()


# In[ ]:


raw_text.replace(r'Page (\d+)\n', '')


# In[ ]:


chars = sorted(list(set(raw_text)))
char_to_int = dict((c,i) for i,c in enumerate(chars))


# In[ ]:


n_chars = len(raw_text)
n_vocab = len(chars)


# In[ ]:


seq_length = 100
dataX = []
dataY = []


# In[ ]:


for i in tqdm(range(0, n_chars - seq_length, 1), desc="Building datasets"):
    seq_in = raw_text[i:i+seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])


# In[ ]:


n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)


# In[ ]:


X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
X = X/float(n_vocab)
y = np_utils.to_categorical(dataY)


# In[ ]:


model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))


# In[ ]:

print("Loading pre-trained weights from weights-improvement-04-1.52475-bigger.hdf5")
model.load_weights("./Checkpoints/weights-improvement-04-1.52475-bigger.hdf5")
model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[ ]:


filepath = "./Checkpoints/weights-improvement-{epoch:02d}-{loss:.5f}-bigger.hdf5"


# In[ ]:


checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


# In[ ]:


model.fit(X, y, epochs=20, batch_size=64, callbacks=callbacks_list, verbose=1)

