import numpy as np
from keras.layers import LSTM, Bidirectional, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import Adam

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
cons_to_int = {char: i for i, char in enumerate(consonants, 1)}
vowel_to_int = {diacritic: i for i, diacritic in enumerate(diacritic_sequences, 1)}

# Reverse mappings
int_to_cons = {i: char for char, i in cons_to_int.items()}
int_to_vowel = {
    i: diacritic for diacritic, i in vowel_to_int.items()
}  # Necessary for decoding predictions

# Convert data
X = [cons_to_int[char] for char, _ in data]
Y = [vowel_to_int[diacritics] for _, diacritics in data]

# Adding sequences
sequence_length = 10
X_sequences = []
Y_sequences = []

for i in range(0, len(X) - sequence_length + 1):
    X_sequences.append(X[i : i + sequence_length])
    Y_sequences.append(Y[i : i + sequence_length])

X_encoded = to_categorical(X_sequences, num_classes=len(cons_to_int) + 1)
Y_encoded = to_categorical(Y_sequences, num_classes=len(vowel_to_int) + 1)

X_encoded = X_encoded.reshape((len(X_encoded), sequence_length, len(cons_to_int) + 1))
Y_encoded = Y_encoded.reshape((len(Y_encoded), sequence_length, len(vowel_to_int) + 1))

# Define model
model = Sequential()
model.add(
    Bidirectional(
        LSTM(64, return_sequences=True, dropout=0.2),
        input_shape=(1, len(cons_to_int) + 1),
    )
)
# model.add(
#     Bidirectional(
#         LSTM(64, return_sequences=True, dropout=0.2),
#         input_shape=(1, len(cons_to_int) + 1),
#     )
# )
model.add(TimeDistributed(Dense(len(vowel_to_int) + 1, activation="softmax")))

# Compile model
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.02),
    metrics=["accuracy"],
)

# Fit model
model.fit(X_encoded, Y_encoded, epochs=1, batch_size=32)

new_text = "بسم الله الرحمن الرحيم قل هو الله احد"

# Assuming 'cons_to_int' mapping is available from your training script
new_sequences = [cons_to_int.get(char, 0) for char in new_text]

# Convert to one-hot encoding
num_classes = len(cons_to_int) + 1  # Include padding class
new_data = to_categorical(new_sequences, num_classes=num_classes)

sequence_length = 10
# Pad the input text with a special character
new_text_padded = new_text + " " * (sequence_length - 1)

new_sequences = []

for i in range(len(new_text_padded) - sequence_length + 1):
    sequence = [
        cons_to_int.get(char, 0) for char in new_text_padded[i : i + sequence_length]
    ]
    new_sequences.append(sequence)

new_data = to_categorical(new_sequences, num_classes=num_classes)

predictions = model.predict(new_data.reshape((-1, sequence_length, num_classes)))

predicted_vowels = [
    int_to_vowel[np.argmax(pred)] for seq in predictions for pred in seq
]

diacritized_phrase = ""
for i in range(len(new_text)):
    cons = new_text[i]
    vowel = predicted_vowels[i] if i < len(predicted_vowels) else "ـ"
    next_vowel = predicted_vowels[i + 1] if i + 1 < len(predicted_vowels) else "ـ"

    if vowel != "ـ":  # Assuming "ـ" is used to denote no diacritic
        diacritized_phrase += cons + vowel
    elif next_vowel == "ـ":
        diacritized_phrase += cons
print(diacritized_phrase)

# # Predict diacritics using the model
# predictions = model.predict(new_data.reshape((-1, 1, num_classes)))

# # Assuming 'int_to_vowel' maps indices back to diacritic sequences
# predicted_vowels = [int_to_vowel[np.argmax(pred)] for pred in predictions]

# # Prepare the predicted diacritized phrase
# diacritized_phrase = ""
# for cons, vowel in zip(new_text, predicted_vowels):
#     if vowel != "ـ":  # Assuming "ـ" is used to denote no diacritic
#         diacritized_phrase += cons + vowel
#     else:
#         diacritized_phrase += cons
# print(diacritized_phrase)
