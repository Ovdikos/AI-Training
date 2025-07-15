import csv
import random
import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.class_labels = {}

    def load_data(self, file_path):
        data = []
        labels = []
        unique_labels = set()

        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if not row:
                    continue
                *features, label = row
                unique_labels.add(label)
                data.append([float(x) for x in features])
                labels.append(label)


        if len(unique_labels) != 2:
            raise ValueError("Dataset must contain exactly 2 classes")
        self.class_labels = {label: idx for idx, label in enumerate(unique_labels)}
        return np.array(data), np.array([self.class_labels[l] for l in labels])

    def initialize_weights(self, n_features):
        self.weights = np.random.rand(n_features)
        self.bias = random.random()

    def predict(self, x):
        net = np.dot(x, self.weights) - self.bias
        return 1 if net >= 0 else 0

    def train(self, X_train, y_train):
        self.initialize_weights(X_train.shape[1])

        for _ in range(self.epochs):
            for x, y in zip(X_train, y_train):
                prediction = self.predict(x)
                error = y - prediction
                self.weights += self.learning_rate * error * x
                self.bias -= self.learning_rate * error

    def evaluate_accuracy(self, X_test, y_test):
        correct = 0
        for x, y in zip(X_test, y_test):
            if self.predict(x) == y:
                correct += 1
        return correct / len(y_test)

    def interactive_mode(self):
        print("\nInteractive mode (enter 'q' to quit):")
        while True:
            user_input = input("Enter features (comma-separated): ").strip()
            if user_input.lower() == 'q':
                break
            try:
                features = [float(x) for x in user_input.split(',')]
                if len(features) != len(self.weights):
                    raise ValueError
                result = self.predict(features)
                class_name = [k for k, v in self.class_labels.items() if v == result][0]
                print(f"Predicted class: {class_name}")
            except:
                print("Invalid input! Expected format: 1.2,3.4,5.6,7.8")


def main():
    perceptron = Perceptron(learning_rate=0.01, epochs=1000)

    try:
        X_train, y_train = perceptron.load_data("perceptron.data")

        perceptron.train(X_train, y_train)

        X_test, y_test = perceptron.load_data("perceptron.test.data")
        accuracy = perceptron.evaluate_accuracy(X_test, y_test)
        print(f"\nTest accuracy: {accuracy:.2%}")

        perceptron.interactive_mode()

    except FileNotFoundError:
        print("Error: Data files not found!")
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()