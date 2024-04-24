def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = [line.strip().split("\t") for line in file if line.strip()]
    return data


def evaluate(test_data, predicted_data):
    correct = 0
    total = 0
    for test, pred in zip(test_data, predicted_data):
        if test[1] == pred[1]:  # Assuming that the diacritic is the second element
            correct += 1
        total += 1
    accuracy = correct / total if total > 0 else 0
    return accuracy


def main():
    # File paths
    test_file_path = "./Data/Parsed Data/test-test-set.txt"
    predicted_file_path = "./src/RNN/predicted-diacritics.txt"

    # Load datasets
    test_data = load_data(test_file_path)
    predicted_data = load_data(predicted_file_path)

    # Evaluate accuracy
    accuracy = evaluate(test_data, predicted_data)
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
