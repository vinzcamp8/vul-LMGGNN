import configs
from utils.data.datamanager import loads, train_val_test_split
import torch
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from torch import nn, optim


PATHS = configs.Paths()
FILES = configs.Files()
DEVICE = FILES.get_device()

context = configs.Process()
input_dataset = loads(PATHS.input)


# # remove samples without edges
# input_dataset = input_dataset[input_dataset['input'].apply(lambda x: x.edge_index.size(1) > 0)]
# # standardize feature vector for handle 0 values in node feature vector
# for id, data in input_dataset.input.items():
#     data.x = (data.x - data.x.mean(dim=0)) / (data.x.std(dim=0) + 1e-6)

# split the dataset and pass to DataLoader with batch size
train_loader, val_loader, test_loader = list(
    map(lambda x: x.get_loader(context.batch_size, shuffle=context.shuffle),
        train_val_test_split(input_dataset, shuffle=context.shuffle)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=== Running Dummy Classifier ===")
# Create a dummy classifier
dummy_clf = DummyClassifier(strategy="stratified")

# print(next(iter(train_loader)))
# DataBatch(x=[1640, 769], edge_index=[2, 1314], y=[8], types=[8], codes=[8], func=[8], batch=[1640], ptr=[9])

# Train the dummy classifier
train_data = []
train_labels = []
for batch in train_loader:
    # Reduce each graph's nodes to a single value by averaging the node features
    graph_representations = torch.zeros((batch.ptr.size(0) - 1, batch.x.size(1)))
    for i in range(batch.ptr.size(0) - 1):
        start, end = batch.ptr[i], batch.ptr[i + 1]
        graph_representations[i] = batch.x[start:end].mean(dim=0)
    train_data.append(graph_representations)
    train_labels.append(batch.y)

train_data = torch.cat(train_data)
train_labels = torch.cat(train_labels)
dummy_clf.fit(train_data, train_labels)

# Prepare validation data
val_data = []
val_labels = []
for batch in val_loader:
    graph_representations = torch.zeros((batch.ptr.size(0) - 1, batch.x.size(1)))
    for i in range(batch.ptr.size(0) - 1):
        start, end = batch.ptr[i], batch.ptr[i + 1]
        graph_representations[i] = batch.x[start:end].mean(dim=0)
    val_data.append(graph_representations)
    val_labels.append(batch.y)

val_data = torch.cat(val_data)
val_labels = torch.cat(val_labels)

# Validate the dummy classifier
val_predictions = dummy_clf.predict(val_data)
val_accuracy = accuracy_score(val_labels, val_predictions)
val_precision = precision_score(val_labels, val_predictions)
val_recall = recall_score(val_labels, val_predictions)
val_f1 = f1_score(val_labels, val_predictions)

print(f"Validation Accuracy: {val_accuracy}")
print(f"Validation Precision: {val_precision}")
print(f"Validation Recall: {val_recall}")
print(f"Validation F1 Score: {val_f1}")

# Compute confusion matrix
val_cm = confusion_matrix(val_labels, val_predictions)
print("Confusion Matrix:")
print(val_cm)

# Prepare test data
test_data = []
test_labels = []
for batch in test_loader:
    graph_representations = torch.zeros((batch.ptr.size(0) - 1, batch.x.size(1)))
    for i in range(batch.ptr.size(0) - 1):
        start, end = batch.ptr[i], batch.ptr[i + 1]
        graph_representations[i] = batch.x[start:end].mean(dim=0)
    test_data.append(graph_representations)
    test_labels.append(batch.y)

test_data = torch.cat(test_data)
test_labels = torch.cat(test_labels)
test_predictions = dummy_clf.predict(test_data)
test_accuracy = accuracy_score(test_labels, test_predictions)
test_precision = precision_score(test_labels, test_predictions)
test_recall = recall_score(test_labels, test_predictions)
test_f1 = f1_score(test_labels, test_predictions)

print(f"Test Accuracy: {test_accuracy}")
print(f"Test Precision: {test_precision}")
print(f"Test Recall: {test_recall}")
print(f"Test F1 Score: {test_f1}")

# Compute confusion matrix
test_cm = confusion_matrix(test_labels, test_predictions)
print("Confusion Matrix:")
print(test_cm)