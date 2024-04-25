import numpy as np
from keras.layers import LSTM, Bidirectional, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical

# Define paths
training_file_path = "./Data/Parsed Data/letter-training.txt"

# Read the training file
with open(training_file_path, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")

# Split each line into a letter and its diacritics
# data = [line.split("\t") for line in lines if line.strip()]
data = []
for line in lines:
    if "\t" in line:
        data.append(line.split("\t"))
    else:
        data.append(["", ""])

# Create mappings
consonants = sorted(set(pair[0] for pair in data))
diacritic_sequences = sorted(set(pair[1] for pair in data))
cons_to_int = {char: i for i, char in enumerate(consonants, 1)}  # Reserve 0 for padding
vowel_to_int = {
    diacritic: i for i, diacritic in enumerate(diacritic_sequences, 1)
}  # Reserve 0 for padding

# Reverse mappings
int_to_cons = {i: char for char, i in cons_to_int.items()}
int_to_vowel = {
    i: diacritic for diacritic, i in vowel_to_int.items()
}  # Necessary for decoding predictions

# Convert data
X = [cons_to_int[char] for char, _ in data]
Y = [vowel_to_int[diacritics] for _, diacritics in data]

# One-hot encoding
X_encoded = to_categorical(X, num_classes=len(cons_to_int) + 1)
Y_encoded = to_categorical(Y, num_classes=len(vowel_to_int) + 1)

# Reshape for LSTM
X_encoded = X_encoded.reshape((len(X_encoded), 1, len(cons_to_int) + 1))
Y_encoded = Y_encoded.reshape((len(Y_encoded), 1, len(vowel_to_int) + 1))

# Define model
model = Sequential()
model.add(
    Bidirectional(
        LSTM(64, return_sequences=True, dropout=0.2),
        input_shape=(1, len(cons_to_int) + 1),
    )
)
model.add(
    Bidirectional(
        LSTM(64, return_sequences=True, dropout=0.2),
        input_shape=(1, len(cons_to_int) + 1),
    )
)
model.add(TimeDistributed(Dense(len(vowel_to_int) + 1, activation="softmax")))

# Compile model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Fit model
model.fit(X_encoded, Y_encoded, epochs=1, batch_size=32)


new_text = "بسم الله الرحمن الرحيم قل هو الله احد"

# Assuming 'cons_to_int' mapping is available from your training script
new_sequences = [cons_to_int.get(char, 0) for char in new_text] 

# Convert to one-hot encoding
num_classes = len(cons_to_int) + 1  # Include padding class
new_data = to_categorical(new_sequences, num_classes=num_classes)

# Predict diacritics using the model
predictions = model.predict(new_data.reshape((-1, 1, num_classes)))

# Assuming 'int_to_vowel' maps indices back to diacritic sequences
predicted_vowels = [int_to_vowel[np.argmax(pred)] for pred in predictions]

# Prepare the predicted diacritized phrase
diacritized_phrase = ""
for cons, vowel in zip(new_text, predicted_vowels):
    if vowel != "ـ":  # Assuming "ـ" is used to denote no diacritic
        diacritized_phrase += cons + vowel
    else:
        diacritized_phrase += cons

# Print the predicted diacritized phrase
print(diacritized_phrase)
