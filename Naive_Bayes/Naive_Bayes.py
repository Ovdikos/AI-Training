import csv
import math


def load_data(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = []
        for row in reader:
            cleaned_row = [item.strip() for item in row]
            if cleaned_row:
                data.append(cleaned_row)
        return data


class NaiveBayesClassifier:
    def __init__(self):
        self.class_counts = {}
        self.feature_counts = {}
        self.unique_values_per_feature = {}
        self.alpha = 1

    def train(self, train_data):
        self.class_counts = {'e': 0, 'p': 0}
        num_features = len(train_data[0]) - 1
        self.feature_counts = {
            'e': [{} for _ in range(num_features)],
            'p': [{} for _ in range(num_features)]
        }
        self.unique_values_per_feature = [set() for _ in range(num_features)]

        for row in train_data:
            label = row[0].lower()
            self.class_counts[label] += 1

            for i in range(num_features):
                value = row[i + 1]
                self.unique_values_per_feature[i].add(value)

                if value in self.feature_counts[label][i]:
                    self.feature_counts[label][i][value] += 1
                else:
                    self.feature_counts[label][i][value] = 1

    def predict(self, test_data):
        predictions = []
        for row in test_data:
            features = row[1:]
            log_probs = {}

            for label in ['e', 'p']:
                total = sum(self.class_counts.values())
                prob_class = (self.class_counts[label] + self.alpha) / (total + self.alpha * 2)
                log_prob = math.log(prob_class)

                for i, value in enumerate(features):
                    count = self.feature_counts[label][i].get(value, 0)
                    num_values = len(self.unique_values_per_feature[i])
                    prob = (count + self.alpha) / (self.class_counts[label] + self.alpha * num_values)
                    log_prob += math.log(prob)

                log_probs[label] = log_prob

            predicted = max(log_probs, key=log_probs.get)
            predictions.append(predicted)

        return predictions


def calculate_metrics(true_labels, predicted_labels):
    tp = fp = tn = fn = 0
    for t, p in zip(true_labels, predicted_labels):
        if t == 'e' and p == 'e':
            tn += 1
        elif t == 'p' and p == 'p':
            tp += 1
        elif t == 'e' and p == 'p':
            fp += 1
        else:
            fn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


train_data = load_data('agaricus-lepiota.data')
test_data = load_data('agaricus-lepiota.test.data')

classifier = NaiveBayesClassifier()
classifier.train(train_data)

true_labels = [row[0].lower() for row in test_data]
predicted_labels = classifier.predict(test_data)

metrics = calculate_metrics(true_labels, predicted_labels)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1-measure: {metrics['f1']:.4f}")