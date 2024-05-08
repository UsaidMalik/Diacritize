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
# max_seq_length = 20
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
        LSTM(32, return_sequences=True, dropout=0.2, dtype="float32"),
        input_shape=(max_seq_length, len(cons_to_int) + 1),
    )
)
model.add(TimeDistributed(Dense(len(vowel_to_int) + 1, activation="softmax")))


# Compile model
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.03),
    metrics=["accuracy"],
)

# Fit model
model.fit(X_encoded, Y_encoded, epochs=1, batch_size=512)

model.save("model.keras", overwrite=True)

# def predict_diacritics(sentence):
#     sentence_cleaned = sentence.replace(" ", "")  # Assuming no spaces needed for internal model processing
#     X_sequence = [cons_to_int.get(char, cons_to_int['<PAD>']) for char in sentence_cleaned]
#     X_padded = X_sequence + [cons_to_int['<PAD>']] * (max_seq_length - len(X_sequence))
#     X_encoded = to_categorical([X_padded], num_classes=len(cons_to_int) + 1)

#     print("Shape of X_encoded:", X_encoded.shape)  # Check input shape

#     predictions = model.predict(X_encoded)[0]
#     print("Predictions:", predictions)  # See what the model is predicting

#     print("max index:", np.argmax(predictions, axis=1))

#     predicted_vowels = [int_to_vowel[np.argmax(pred)] if np.argmax(pred) in int_to_vowel else "" for pred in predictions[:len(sentence_cleaned)]]
#     print("Predicted Diacritics:", predicted_vowels)  # Check the diacritics being applied

#     diacritized_sentence = ""
#     idx = 0
#     for char in sentence:
#         if char == " ":
#             diacritized_sentence += " "
#         else:
#             diacritized_sentence += char + predicted_vowels[idx]
#             idx += 1

#     return diacritized_sentence


# # Example usage
# new_text = "باربي هي رغبة رائعة للأطفال	باربي هي الرغبة الرهيبة للأطفال"
# predicted_text = predict_diacritics(new_text)
# print(predicted_text)


def diacritize_text(text):
    # Convert text into sequences
    X_sequence = [cons_to_int.get(char, 0) for char in text]

    # Pad sequence
    max_seq_length = model.input_shape[1]
    X_sequence = X_sequence + [padding_token] * (max_seq_length - len(X_sequence))

    # Reshape and encode input
    X_encoded = to_categorical([X_sequence], num_classes=len(cons_to_int) + 1)

    # Predict diacritics
    Y_pred = model.predict(X_encoded)

    # Decode predicted diacritics
    Y_decoded = np.argmax(Y_pred, axis=-1)[0]
    diacritics = [int_to_vowel.get(i, "") for i in Y_decoded]

    # Combine consonants and diacritics
    diacritized_text = ""
    for char, diacritic in zip(text, diacritics):
        diacritized_text += char + diacritic

    return diacritized_text


# Example usage
new_text = "يحب التنزه في الحديقة مع سريره"
diacritized_text = diacritize_text(new_text)
print("Diacritized text:", diacritized_text)