from string import ascii_lowercase
import csv


class SingleLayerNN:
    def __init__(self, classes_num: int, weights_len: int, alpha: float, epochs: int):
        self.classes_num = classes_num
        self.weights_len = weights_len
        self.alpha = alpha
        self.epochs = epochs
        self.initialize()

    def initialize(self):
        self.weights = [
            [0 for _ in range(self.weights_len)]
            for _ in range(self.classes_num)
        ]
        self.biases = [0 for _ in range(self.classes_num)]

    #Makes the sum of the squares of the vector elements = 1
    @staticmethod
    def normalize_vector(vector: list):
        length = sum(v * v for v in vector) ** .5
        return list(map(lambda v: v / length, vector))

    #Caclulating how many times me met some letter, normalizing vector
    @classmethod
    def vectorize(cls, text):
        vector = [0 for _ in range(len(ascii_lowercase))]

        for c in text.lower():
            if c not in ascii_lowercase:
                continue

            vector[ascii_lowercase.index(c)] += 1

        return cls.normalize_vector(vector)

    #Scalar
    def dot_product(self, vector1, vector2):
        return sum(vector1[i] * vector2[i] for i in range(self.weights_len))

    #Sum of vectors
    def add_vectors(self, vector1, vector2):
        return list(vector1[i] + vector2[i] for i in range(self.weights_len))

    def get_activation(self, class_, vector):
        return self.dot_product(self.weights[class_], vector) - self.biases[class_]

    #Predictions for all languages
    def get_predictions(self, vector: list):
        predictions = []

        for i in range(self.classes_num):
            predictions.append(self.get_activation(i, vector))

        return predictions

    #For each language update scales and bias
    def update_weights(self, vector: list, desired_class: int):
        for i in range(self.classes_num):
            class_weights = self.weights[i]
            activation = self.get_activation(i, vector)
            desired_value = int(desired_class == i)
            self.weights[i] = self.add_vectors(
                class_weights, [self.alpha * (desired_value - activation) * x for x in vector]
            )
            self.biases[i] -= self.alpha * (desired_value - activation)

    def get_predictions_for_text(self, text):
        vector = self.vectorize(text)
        return self.get_predictions(vector)

    #For each text from train-data calculates the vector and updates the scales
    def train(self, train_texts: list, train_labels: list, labels: list):
        for i in range(len(train_texts)):
            text = train_texts[i]
            label = train_labels[i]
            label_index = labels.index(label)
            data_vector = self.vectorize(text)
            self.update_weights(data_vector, label_index)

    def test(self, test_texts: list, test_labels: list, labels: list):
        correct = 0
        for text, label in zip(test_texts, test_labels):
            predicted_label = self.get_predicted_label(text, labels)

            if label == predicted_label:
                correct += 1

        return correct / len(test_labels)

    def get_predicted_label(self, text, labels):
        vector = self.vectorize(text)
        predictions = self.get_predictions(vector)
        predicted_label = labels[predictions.index(max(predictions))]
        return predicted_label


TRAIN_SET_PATH = 'lang.train.csv'
TEST_SET_PATH = 'lang.test.csv'


def get_set(path):
    data = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    return data


def get_labels(data):
    return [d[0] for d in data]


def get_text(data):
    return [d[1] for d in data]


train_data = get_set(TRAIN_SET_PATH)
test_data = get_set(TEST_SET_PATH)

train_labels = get_labels(train_data)
test_labels = get_labels(test_data)

train_texts = get_text(train_data)
test_texts = get_text(test_data)

labels = list(set(train_labels))
classes_num = len(labels)
weights_len = len(ascii_lowercase)
nn = SingleLayerNN(classes_num, weights_len, 0.1, 100_000)

nn.train(train_texts, train_labels, labels)
print('accuracy', nn.test(test_texts, test_labels, labels))

while True:
    text = input("Enter text to predict language: ")
    print('Your predicted language is:', nn.get_predicted_label(text, labels))