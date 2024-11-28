import configs
import torch
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import os


PATHS = configs.Paths()

def run_dummy(train_loader, val_loader, test_loader):
    strategy = "stratified"
    print(f'=== Running Dummy Classifier with strategy "{strategy}" ===')
    # Create a dummy classifier
    dummy_clf = DummyClassifier(strategy=strategy)

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

    print(f"Validation - Accuracy: {val_accuracy}, Precision: {val_precision}, Recall: {val_recall}, F1: {val_f1}")
    

    # Compute confusion matrix
    val_cm = confusion_matrix(val_labels, val_predictions)
    print("Validation Confusion Matrix:")
    print(val_cm)

    path_results = f"{PATHS.model}dummy/"
    os.makedirs(path_results)
    with open(f"{path_results}val_metrics.txt", 'a') as f:
        f.write(f"Validation - Accuracy: {val_accuracy}, Precision: {val_precision}, Recall: {val_recall}, F1: {val_f1}")
        f.write("\nValidation Confusion Matrix:")
        f.write("\n")
        f.write(str(val_cm))
    
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

    print(f"Test - Accuracy: {test_accuracy}, Precision: {test_precision}, Recall: {test_recall}, F1: {test_f1}")

    # Compute confusion matrix
    test_cm = confusion_matrix(test_labels, test_predictions)
    print("Test Confusion Matrix:")
    print(test_cm)

    with open(f"{path_results}test_metrics.txt", 'a') as f:
        f.write(f"Test - Accuracy: {test_accuracy}, Precision: {test_precision}, Recall: {test_recall}, F1: {test_f1}")
        f.write("\nTest Confusion Matrix:")
        f.write("\n")
        f.write(str(test_cm))