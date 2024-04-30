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
data = []
for line in lines:
    if line.strip() == "ENDAYAT":
        data.append(["ENDAYAT", ""])
    elif "\t" in line:
        data.append(line.split("\t"))
    else:
        data.append(["", ""])

# Create mappings
consonants = sorted(set(pair[0] for pair in data))
diacritic_sequences = sorted(set(pair[1] for pair in data))
cons_to_int = {char: i for i, char in enumerate(consonants, 1)}
vowel_to_int = {diacritic: i for i, diacritic in enumerate(diacritic_sequences, 1)}

# Add padding token to cons_to_int
padding_token = len(cons_to_int) + 1
cons_to_int["<PAD>"] = padding_token

# Reverse mappings
int_to_cons = {i: char for char, i in cons_to_int.items()}
int_to_vowel = {i: diacritic for diacritic, i in vowel_to_int.items()}

# Convert data into sequences
max_seq_length = max(len(pair[0]) for pair in data)
X_sequences = []
Y_sequences = []

for consonant, diacritic in data:
    if consonant == "ENDAYAT":
        continue
    X_sequence = [cons_to_int.get(char, 0) for char in consonant]
    Y_sequence = [vowel_to_int.get(char, 0) for char in diacritic]

    # Pad sequences
    X_sequence = X_sequence + [padding_token] * (max_seq_length - len(X_sequence))
    Y_sequence = Y_sequence + [0] * (max_seq_length - len(Y_sequence))

    X_sequences.append(X_sequence)
    Y_sequences.append(Y_sequence)

X_encoded = to_categorical(X_sequences, num_classes=len(cons_to_int) + 1)
Y_encoded = to_categorical(Y_sequences, num_classes=len(vowel_to_int) + 1)

# Define model
model = Sequential()
model.add(
    Bidirectional(
        LSTM(64, return_sequences=True, dropout=0.2),
        input_shape=(max_seq_length, len(cons_to_int) + 1),
    )
)
model.add(TimeDistributed(Dense(len(vowel_to_int) + 1, activation="softmax")))

# Compile model
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.02),
    metrics=["accuracy"],
)

# Fit model
model.fit(X_encoded, Y_encoded, epochs=1, batch_size=64)

def predict_diacritics(text):
    diacritized_words = []
    words = text.split()  # Split the input text into words

    for word in words:
        X_sequence = [cons_to_int.get(char, 0) for char in word]
        X_sequence = X_sequence + [padding_token] * (max_seq_length - len(X_sequence))
        X_encoded = to_categorical([X_sequence], num_classes=len(cons_to_int) + 1)

        predictions = model.predict(X_encoded)[0]
        predicted_vowels = []
        for pred in predictions:
            vowel_index = np.argmax(pred)
            if vowel_index in int_to_vowel:
                predicted_vowels.append(int_to_vowel[vowel_index])
            else:
                predicted_vowels.append("")

        diacritized_word = ""
        for i, cons in enumerate(word):
            vowel = predicted_vowels[i] if i < len(predicted_vowels) else ""
            diacritized_word += cons + vowel

        diacritized_words.append(diacritized_word)

    return " ".join(diacritized_words)

# Example usage
new_text = "بسم الله الرحمن الرحيم"
predicted_text = predict_diacritics(new_text)
print(predicted_text)