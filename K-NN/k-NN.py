import csv
from collections import Counter

class KNN:

    def __init__(self, k):
        self.k = k
        self.train_data = []
        self.train_labels = []

    def load_data(self, file_path):
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row:
                    *features, label = row
                    self.train_data.append([float(x) for x in features])
                    self.train_labels.append(label)

    def euclidean_distance(self, a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b))


    def predict(self, test_point):
        distances = [
            (self.euclidean_distance(test_point, point), label)
            for point, label in zip(self.train_data, self.train_labels)
        ]
        neighbors = sorted(distances, key=lambda x: x[0])[:self.k]
        return Counter(label for (_, label) in neighbors).most_common(1)[0][0]

    def test_accuracy(self, test_file):
        correct = 0
        total = 0
        with open(test_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row:
                    *features, true_label = row
                    prediction = self.predict([float(x) for x in features])
                    if prediction == true_label:
                        correct += 1
                    total += 1
        return correct / total if total else 0.0

    def run_interface(self, test_file):

        while True:
            choice = input("\nSelect mode: \n"
                           "1. Classify single vector \n"
                           "2. Test accuracy using predefined test file \n"
                           "3. Exit \n").strip()

            if choice == '1':
                self._single_vector_mode()
            elif choice == '2':
                accuracy = self.test_accuracy(test_file)
                print(f"\nAccuracy on test set: {accuracy:.2%}")
            elif choice == '3':
                print("Exiting...")
                break
            else:
                print("Invalid input. Please enter 1, 2 or 3")
                print(choice)

    def _single_vector_mode(self):
        try:
            vector = [float(x) for x in input("Enter vector (comma-separated): ").split(',')]
            print(f"Predicted class: {self.predict(vector)}")
        except:
            print("Error: Invalid input format")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='k-NN Classifier')
    parser.add_argument('k', type=int, help='Number of neighbors')
    parser.add_argument('train_set', help='Path to training dataset')
    parser.add_argument('test_set', help='Path to test dataset')
    args = parser.parse_args()

    knn = KNN(args.k)
    knn.load_data(args.train_set)
    knn.run_interface(args.test_set)