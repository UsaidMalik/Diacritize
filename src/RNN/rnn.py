import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

training_file_path = './data/Parsed Data/quran-letter.txt'

# Read the training file
with open(training_file_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

# Split each line into a letter and a mark
data = [line.split('\t') for line in lines if line]

# Create mappings
char_to_int = {char: i for i, char in enumerate(sorted(set(char for pair in data for char in pair)))}
int_to_char = {i: char for char, i in char_to_int.items()}

# Convert data to sequences of integers
X = [[char_to_int[char] for char in pair] for pair in data]

# Separate letters and marks
letters = [pair[0] for pair in X]
marks = [pair[1] for pair in X]

# Convert to one-hot encoding
letters = to_categorical(letters, num_classes=len(char_to_int))
marks = to_categorical(marks, num_classes=len(char_to_int))

# Define your model
model = Sequential()
model.add(Embedding(input_dim=len(char_to_int)+1, output_dim=64, input_length=letters.shape[1]))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(len(char_to_int), activation='softmax'))

# Compile and train your model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(letters, marks, epochs=1, batch_size=64)

# Predict marks for new text
new_text = "اللغة العربية"
new_sequences = [char_to_int[char] for char in new_text]
new_data = pad_sequences([new_sequences], maxlen=letters.shape[1])
predictions = model.predict(new_data)
print(''.join(int_to_char[i] for i in np.argmax(predictions, axis=1)))