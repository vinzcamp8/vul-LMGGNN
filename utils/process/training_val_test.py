import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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
    if (model.__class__.__name__ == 'BertGGCN'):
        vul_lmgnn_flag = True
    else:
        vul_lmgnn_flag = False
    
    model.train()

    for batch_idx, batch in enumerate(train_loader):
        batch.to(device)

        # vul_lmgnn need the entire batch cause use other features like "func"
        y_pred = model(batch) if vul_lmgnn_flag else model(batch.x, batch.edge_index, batch.batch)
        
        model.zero_grad()
  
        batch.y = batch.y.squeeze().long() 
        # batch.y = batch.y.long() # (debugging) if batch_size = 1 

        loss = F.cross_entropy(y_pred, batch.y)
        loss.backward()
        optimizer.step()
        
        # (debugging)
        if (batch_idx + 1) % 100 == 0: # print every 100 mini-batches
            print('=== Train Epoch: {} [{}/{} ({:.2f}%)]/t Loss: {:.6f}'.format(epoch, (batch_idx + 1) * len(batch),
                                                                            len(train_loader.dataset),
                                                                            100. * batch_idx / len(train_loader),
                                                                            loss.item()))
            print("batch y_pred min/max/mean: ", y_pred.min().item(), y_pred.max().item(), y_pred.mean().item())        


def validate(model, device, test_loader, path_output_results, epoch):
    """
    Validates the model using the provided test data.

    :param model: The model to be validated.
    :param device: The device to perform validation on (e.g., 'cpu' or 'cuda').
    :param test_loader: The data loader containing the test data.
    :return: Tuple containing accuracy, precision, recall, and F1 score.
    """
    if (model.__class__.__name__ == 'BertGGCN'):
        vul_lmgnn_flag = True
    else:
        vul_lmgnn_flag = False

    model.eval()
    test_loss = 0.0
    y_true = []
    y_pred = []

    for batch_idx, batch in enumerate(test_loader):
        batch.to(device)
        with torch.no_grad():
            # vul_lmgnn need the entire batch cause use other features like "func"
            y_ = model(batch) if vul_lmgnn_flag else model(batch.x, batch.edge_index, batch.batch)

        batch.y = batch.y.squeeze().long()
        # batch.y = batch.y.long() # (debugging) if batch_size = 1 
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
    print('Validation Confusion Matrix:')
    print(cm)

    # print('Validation set - Average loss: {:.4f}, Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(
    #     test_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100))

    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['benign', 'malware'], yticklabels=['benign', 'malware'])
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # plt.savefig('confusion_matrix.png')

    with open(f'{path_output_results}{model.__class__.__name__}_val_metrics.txt', 'a') as f:
        f.write(f'\n=== Epoch {epoch} ===\n\nValidation Confusion Matrix:\n')
        np.savetxt(f, cm, fmt='%d', delimiter='\t')
        f.write('\n')
        f.write('Validation set: Average loss: {:.4f}, Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%\n'.format(
        test_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100))

    return accuracy, precision, recall, f1

def test(model, device, test_loader, path_output_results):
    """
    Tests the model using the provided test data loader.

    :param model: The model to be tested.
    :param device: The device to perform testing on (e.g., 'cpu' or 'cuda').
    :param test_loader: The data loader containing the test data.
    :return: Tuple containing accuracy, precision, recall, and F1 score.
    """
    if (model.__class__.__name__ == 'BertGGCN'):
        vul_lmgnn_flag = True
    else:
        vul_lmgnn_flag = False

    model.eval()
    test_loss = 0.0
    y_true = []
    y_pred = []
    y_probs = []

    for batch_idx, batch in enumerate(test_loader):
        batch.to(device)
        with torch.no_grad():
            # vul_lmgnn need the entire batch cause use other features like "func"
            y_ = model(batch) if vul_lmgnn_flag else model(batch.x, batch.edge_index, batch.batch)

        batch.y = batch.y.squeeze().long()
        # batch.y = batch.y.long() # (debugging) if batch_size = 1
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

    print('Test set - Average loss: {:.4f}, Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(
        test_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100))

    cm = confusion_matrix(y_true, y_pred)
    print("Test Confusion Matrix: ")
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
    with open(f'{path_output_results}{model.__class__.__name__}_test_probabilities.txt', 'a') as f:
        np.savetxt(f, results_array, fmt='%1.6f', delimiter='\t', header=header_text)
    
    with open(f'{path_output_results}{model.__class__.__name__}_test_metrics.txt', 'a') as f:
        f.write('Test Confusion Matrix:\n')
        np.savetxt(f, cm, fmt='%d', delimiter='\t')
        f.write('\n')
        f.write('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%\n'.format(
            test_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100))
    
    return accuracy, precision, recall, f1

'''
How to read Confusion Matrix for binary classification
[[TN   FP]
 [ FN   TP]]
'''