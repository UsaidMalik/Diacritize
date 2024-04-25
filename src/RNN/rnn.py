import numpy as np
from keras.layers import LSTM, Bidirectional, Dense
from keras.models import Sequential
from keras.utils import to_categorical

training_file_path = "./Data/Parsed Data/test-training-set.txt"

# Read the training file
with open(training_file_path, "r", encoding="utf-8") as f:
    test_lines = f.read().split("\n")

# Split each line into a letter and a mark
data = [line.split("\t") for line in test_lines if line]


# Check the data
print(data[:5])

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
new_text = "بسم الله الرحمن الرحيم قل هو الله احمد"
new_sequences = [cons_to_int[char] for char in new_text]
new_data = to_categorical([new_sequences], num_classes=num_classes)
predictions = model.predict(new_data.reshape((-1, 1, num_classes)))
predicted_vowels = [int_to_vowel[i] for i in np.argmax(predictions, axis=1)]

# Print the predicted diacritized phrase
diacritized_phrase = ""
for cons, vowel in zip(new_text, predicted_vowels):
    # diacritized_phrase += cons + vowel
    # add nothing if the vowel is empty
    if vowel != "ـ":
        diacritized_phrase += cons + vowel
    else:
        diacritized_phrase += cons
print(diacritized_phrase)


# Test the model
# testing_file_path = "./Data/Parsed Data/test-test-set.txt"

# # Read the testing file
# with open(testing_file_path, "r", encoding="utf-8") as f:
#     test_lines = f.readlines()

# # Split each line into a letter and mark (assuming the format is consistent with the training data)
# test_data = [line.split("\t") for line in test_lines if line]

# # Separate letters and convert them to indices using the existing dictionary
# test_consonants = [
#     cons_to_int.get(pair[0], 0) for pair in test_data
# ]  # Use 0 or another index for unseen characters

# # Convert test letters to one-hot encoding
# test_consonants = to_categorical(test_consonants, num_classes=len(cons_to_int))

# # Predict diacritics using the model
# predictions = model.predict(test_consonants.reshape((-1, 1, len(cons_to_int))))
# predicted_indices = np.argmax(predictions, axis=1)
# predicted_vowels = [int_to_vowel[i] for i in predicted_indices]

# # Prepare the output data format
# output_data = [
#     f"{cons}\t{vowel}"
#     for cons, vowel in zip([pair[0] for pair in test_data], predicted_vowels)
# ]

# # Write the predicted data to a file
# output_file_path = "./src/RNN/predicted-diacritics.txt"
# with open(output_file_path, "w", encoding="utf-8") as f:
#     for line in output_data:
#         f.write(line + "\n")
