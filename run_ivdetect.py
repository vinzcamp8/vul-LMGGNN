import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import configs
import torch
import torch.nn.functional as F
from utils.data.datamanager import loads, train_val_test_split
from models.LMGNN import BertGGCN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

def train(model, device, train_loader, optimizer, epoch):
    """
    Trains the model using the provided data.

    :param model: The model to be trained.
    :param device: The device to perform training on (e.g., 'cpu' or 'cuda').
    :param train_loader: The data loader containing the training data.
    :param optimizer: The optimizer used for training.
    :param epoch: The current epoch number.
    :return: None
    """

    model.train()
    best_acc = 0.0
    for batch_idx, batch in enumerate(train_loader):
        batch.to(device)

        y_pred = model(batch.x, batch.edge_index, batch.batch)
        model.zero_grad()

        # print("=== in train() y_pred min/max: ", y_pred.min().item(), y_pred.max().item())
  
        batch.y = batch.y.squeeze().long() ### CODICE ORIGINALE
#         batch.y = batch.y.long()
        
        loss = F.cross_entropy(y_pred, batch.y)
        loss.backward()
        optimizer.step()
        # print(f"=== LOSS in train() backward: {loss}")
        
#         if (batch_idx + 1) % 100 == 0:
    print('Train Epoch: {} [{}/{} ({:.2f}%)]/t Loss: {:.6f}'.format(epoch, (batch_idx + 1) * len(batch),
                                                                            len(train_loader.dataset),
                                                                            100. * batch_idx / len(train_loader),
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
            y_ = model(batch.x, batch.edge_index, batch.batch)

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

    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['benign', 'malware'], yticklabels=['benign', 'malware'])
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # plt.savefig('confusion_matrix.png')

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

    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['benign', 'malware'],
    #             yticklabels=['benign', 'malware'])
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # plt.savefig('confusion_matrix.png')

    # y_true = np.nan_to_num(0)
    # y_probs = np.nan_to_num(0)
    # fpr, tpr, _ = roc_curve(y_true, y_probs)
    # roc_auc = auc(fpr, tpr)

    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    # plt.legend(loc='lower right')
    # plt.savefig('roc_curve.png')

    results_array = np.column_stack((y_true, y_pred, y_probs))
    header_text = "True label, Predicted label, Predicted Probability"
    np.savetxt('ivdetect_results.txt', results_array, fmt='%1.6f', delimiter='\t', header=header_text)

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

from models.ivdetect import IVDetectModel

# Initialize the model and optimizer
model = IVDetectModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.00001)
# Train the model
best_acc = 0.0
NUM_EPOCHS = 3
PATH = "data/model/vinz_ivdetect_model.pth"
for epoch in range(1, NUM_EPOCHS + 1):
    train(model, device, train_loader, optimizer, epoch)
    acc, precision, recall, f1 = validate(model, DEVICE, val_loader)
    if best_acc <= acc:
        best_acc = acc
        torch.save(model.state_dict(), PATH)
    print("====== Acc is: {:.4f}, best acc is {:.4f}".format(acc, best_acc))

# Test the model
model_test = IVDetectModel().to(device)
model_test.load_state_dict(torch.load(PATH))
accuracy, precision, recall, f1 = test(model_test, DEVICE, test_loader)