import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import configs
import torch
import torch.nn.functional as F
from utils.data.datamanager import loads, train_val_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        batch.to(device)

        y_pred = model(batch.x, batch.edge_index, batch.batch)  # Graph-level prediction
        model.zero_grad()

        batch.y = batch.y.squeeze().long()
        loss = F.cross_entropy(y_pred, batch.y)  # Now batch_size matches
        loss.backward()
        optimizer.step()

    print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
        epoch, (batch_idx + 1) * len(batch),
        len(train_loader.dataset), 100. * batch_idx / len(train_loader),
        loss.item()))



def validate(model, device, test_loader):
    """
    Validates the model using the provided test data.

    :param model: The model to be validated.
    :param device: The device to perform validation on (e.g., 'cpu' or 'cuda').
    :param test_loader: The data loader containing the test data.
    :return: Tuple containing accuracy, precision, recall, and F1 score.
    """
    model.eval()
    test_loss = 0.0
    y_true = []
    y_pred = []

    for batch_idx, batch in enumerate(test_loader):
        batch.to(device)
        with torch.no_grad():
            y_ = model(batch.x, batch.edge_index, batch.batch)  # Graph-level prediction

        # batch.y = batch.y.squeeze().long()
        batch.y = batch.y.long()
        test_loss += F.cross_entropy(y_, batch.y).item()
        pred = y_.max(-1, keepdim=True)[1]

        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

    try:
        test_loss /= len(test_loader)
    except ZeroDivisionError:
        test_loss = 0.0

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    print("=== Validation confusion matrix: ")
    print(cm)

    print('Validation set: Average loss: {:.4f}, Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(
        test_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100))

    return accuracy, precision, recall, f1

def test(model, device, test_loader):
    """
    Tests the model using the provided test data loader.

    :param model: The model to be tested.
    :param device: The device to perform testing on (e.g., 'cpu' or 'cuda').
    :param test_loader: The data loader containing the test data.
    :return: Tuple containing accuracy, precision, recall, and F1 score.
    """
    model.eval()
    test_loss = 0.0
    y_true = []
    y_pred = []
    y_probs = []

    for batch_idx, batch in enumerate(test_loader):
        batch.to(device)
        with torch.no_grad():
            y_ = model(batch.x, batch.edge_index, batch.batch)

        # batch.y = batch.y.squeeze().long()
        batch.y = batch.y.long()
        test_loss += F.cross_entropy(y_, batch.y).item()

        pred = y_.max(-1, keepdim=True)[1]
        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
        y_probs.extend(torch.softmax(y_, dim=1).cpu().numpy()[:, 1])

    test_loss /= len(test_loader)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(
        test_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100))

    cm = confusion_matrix(y_true, y_pred) 
    print("=== Test confusion matrix: ")
    print(cm)

    results_array = np.column_stack((y_true, y_pred, y_probs))
    header_text = "True label, Predicted label, Predicted Probability"
    np.savetxt('reveal_results.txt', results_array, fmt='%1.6f', delimiter='\t', header=header_text)

    print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(
        test_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100))

    return accuracy, precision, recall, f1

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

import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU
import torch_geometric.nn as pyg_nn

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLP, self).__init__()
        self.model = Sequential(
            Linear(input_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, num_classes)
        )
        self.pool = pyg_nn.global_mean_pool  # Replace with desired pooling method

    def forward(self, x, edge_index=None, batch=None):
        x = self.model(x)
        if batch is not None:
            x = self.pool(x, batch)  # Pool node-level features to graph-level
        return x


# Set the input dimensions for the MLP
INPUT_DIM = train_loader.dataset[0].x.size(1)  # Feature vector size
HIDDEN_DIM = 128  # Hidden layer size
NUM_CLASSES = len(torch.unique(torch.cat([data.y for data in train_loader.dataset])))

# Initialize the model, optimizer, and loss function
mlp_model = MLP(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES).to(device)
mlp_optimizer = torch.optim.AdamW(mlp_model.parameters(), lr=0.0001, weight_decay=0.00001)

# Training loop for MLP
BEST_ACC = 0.0
MLP_PATH = "data/model/mlp_baseline.pth"
NUM_EPOCHS = 3

for epoch in range(1, NUM_EPOCHS + 1):
    train(mlp_model, device, train_loader, mlp_optimizer, epoch)
    acc, precision, recall, f1 = validate(mlp_model, DEVICE, val_loader)
    if BEST_ACC <= acc:
        BEST_ACC = acc
        torch.save(mlp_model.state_dict(), MLP_PATH)
    print("====== MLP Acc: {:.4f}, Best MLP Acc: {:.4f}".format(acc, BEST_ACC))

# Test the MLP model
mlp_test = MLP(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES).to(device)
mlp_test.load_state_dict(torch.load(MLP_PATH))
accuracy, precision, recall, f1 = test(mlp_test, DEVICE, test_loader)
