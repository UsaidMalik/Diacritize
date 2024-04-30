import numpy as np
from keras.models import load_model
from keras.utils import to_categorical

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


# Define paths
model_path = "model.keras"

# Load the saved model
loaded_model = load_model(model_path)


# Add padding token to cons_to_int
padding_token = len(cons_to_int) + 1
cons_to_int["<PAD>"] = padding_token

# Reverse mappings
int_to_cons = {i: char for char, i in cons_to_int.items()}
int_to_vowel = {i: diacritic for diacritic, i in vowel_to_int.items()}

def diacritize_text(text):
    # Convert text into sequences
    X_sequence = [cons_to_int.get(char, 0) for char in text]
    
    # Pad sequence
    max_seq_length = loaded_model.input_shape[1]
    X_sequence = X_sequence + [padding_token] * (max_seq_length - len(X_sequence))
    
    # Reshape and encode input
    X_encoded = to_categorical([X_sequence], num_classes=len(cons_to_int) + 1)
    
    # Predict diacritics
    Y_pred = loaded_model.predict(X_encoded)
    
    # Decode predicted diacritics
    Y_decoded = np.argmax(Y_pred, axis=-1)[0]
    diacritics = [int_to_vowel.get(i, "") for i in Y_decoded]
    
    # Combine consonants and diacritics
    diacritized_text = ""
    for char, diacritic in zip(text, diacritics):
        diacritized_text += char + diacritic
    
    return diacritized_text

# Example usage
new_text = "باربي هي رغبة رائعة للأطفال باربي هي الرغبة الرهيبة للأطفال"
diacritized_text = diacritize_text(new_text)
print("Diacritized text:", diacritized_text)