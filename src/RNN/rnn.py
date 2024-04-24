import numpy as np
from keras.layers import LSTM, Bidirectional, Dense
from keras.models import Sequential
from keras.utils import to_categorical

training_file_path = "./Data/Parsed Data/test-training-set.txt"

# Read the training file
with open(training_file_path, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")

# Split each line into a letter and a mark
data = [line.split("\t") for line in lines if line]

# Create mappings
consonants = sorted(set(pair[0] for pair in data))
vowels = sorted(set(pair[1] for pair in data))
cons_to_int = {cons: i for i, cons in enumerate(consonants)}
vowel_to_int = {vowel: i for i, vowel in enumerate(vowels)}
int_to_cons = {i: cons for cons, i in cons_to_int.items()}
int_to_vowel = {i: vowel for vowel, i in vowel_to_int.items()}

# Separate letters and marks
consonants = [cons_to_int[pair[0]] for pair in data]
vowels = [vowel_to_int[pair[1]] for pair in data]

# Convert letters to one-hot encoding
num_classes = len(cons_to_int)
consonants = to_categorical(consonants, num_classes=num_classes)

# Define your model
model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(1, num_classes)))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(len(vowel_to_int), activation="softmax"))

# Compile and train your model
model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)
model.fit(
    consonants.reshape((-1, 1, num_classes)), np.array(vowels), epochs=1, batch_size=32
)

# Predict marks for new text
new_text = "فإن لم يكونا كذلك أتى بما يقتضيه الحال وهذا أولى"
new_sequences = [cons_to_int[char] for char in new_text]
new_data = to_categorical([new_sequences], num_classes=num_classes)
predictions = model.predict(new_data.reshape((-1, 1, num_classes)))
predicted_vowels = [int_to_vowel[i] for i in np.argmax(predictions, axis=1)]

# Print the predicted diacritized phrase
diacritized_phrase = ""
for cons, vowel in zip(new_text, predicted_vowels):
    diacritized_phrase += cons + vowel
print(diacritized_phrase)
