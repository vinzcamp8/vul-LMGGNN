import argparse
import gc
import shutil
from argparse import ArgumentParser
import configs
import utils.data as data
import utils.process as process
import utils.functions.cpg_mod as cpg
import torch
import torch.nn.functional as F
from utils.data.datamanager import loads, train_val_test_split
from models.LMGNN import BertGGCN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from test import test
from utils.functions.input_dataset import InputDataset

PATHS = configs.Paths()
FILES = configs.Files()
DEVICE = FILES.get_device()


def select(dataset):
    # dataset = dataset.loc[dataset['project'] == "FFmpeg"]
    dataset = dataset.loc[dataset.func.str.len() < 2000]
    # dataset = dataset.head(100)
    return dataset

def CPG_generator():
    """
    Generates Code Property Graph (CPG) datasets from raw data.

    :return: None
    """
    context = configs.Create()
    raw = data.read(PATHS.raw, FILES.raw)

    # Here, taking the Devign dataset as an example,
    # specific modifications need to be made according to different dataset formats.
    print(f"=== Raw dataset size: {len(raw)} ===")
    filtered = data.apply_filter(raw, select) # see select function above
    print(f"=== Filtered dataset size: {len(filtered)} ===")
    filtered = data.clean(filtered) # remove duplicates 
    print(f"=== Filtered and cleaned dataset size: {len(filtered)} ===")
    data.drop(filtered, ["commit_id", "project"])
    slices = data.slice_frame(filtered, context.slice_size)
    slices = [(s, slice.apply(lambda x: x)) for s, slice in slices]

    cpg_files = []
    # Create CPG binary files
    for s, slice in slices:
        data.to_files(slice, PATHS.joern)
        cpg_file = process.joern_parse(context.joern_cli_dir, PATHS.joern, PATHS.cpg, f"{s}_{FILES.cpg}")
        cpg_files.append(cpg_file)
        print(f"Dataset {s} to cpg.")
        shutil.rmtree(PATHS.joern)
    # Create CPG with graphs json files
    json_files = process.joern_create(context.joern_cli_dir, PATHS.cpg, PATHS.cpg, cpg_files)
    for (s, slice), json_file in zip(slices, json_files):
        graphs = process.json_process(PATHS.cpg, json_file)
        if graphs is None:
            print(f"Dataset chunk {s} not processed.")
            continue
        dataset = data.create_with_index(graphs, ["Index", "cpg"])
        dataset = data.inner_join_by_index(slice, dataset)
        print(f"Writing cpg dataset chunk {s}.")
        data.write(dataset, PATHS.cpg, f"{s}_{FILES.cpg}.pkl")
        del dataset
        gc.collect()

def Embed_generator():
    """
    Generates embeddings from Code Property Graph (CPG) datasets.

    :return: None
    """
    context = configs.Embed()
    dataset_files = data.get_directory_files(PATHS.cpg)

    for pkl_file in dataset_files:
        file_name = pkl_file.split(".")[0]
        cpg_dataset = data.load(PATHS.cpg, pkl_file)
        cpg_dataset = cpg_dataset.head(3)
        print(f"=== Processing input dataset {file_name} with size {len(cpg_dataset)}. ===")
        tokens_dataset = data.tokenize(cpg_dataset)
        data.write(tokens_dataset, PATHS.tokens, f"{file_name}_{FILES.tokens}")

        cpg_dataset["nodes"] = cpg_dataset.apply(lambda row: cpg.parse_to_nodes(row.cpg, context.nodes_dim), axis=1)
        cpg_dataset["input"] = cpg_dataset.apply(lambda row: process.nodes_to_input(row.nodes, row.target, context.nodes_dim,
                                                                            context.edge_type), axis=1)
        data.drop(cpg_dataset, ["nodes"])
        print(f"=== Saving input dataset {file_name} with size {len(cpg_dataset)}. ===")
        print(f"=== Saving input dataset {cpg_dataset['input']}. ===")

        print(f"=== First element of input dataset: {cpg_dataset['input'].iloc[0]} ===")


        data.write(cpg_dataset[["input", "target", "func"]], PATHS.input, f"{file_name}_{FILES.input}")

        del cpg_dataset
        gc.collect()

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

        y_pred = model(batch)
        model.zero_grad()

        print("=== in train() y_pred min/max: ", y_pred.min().item(), y_pred.max().item())
  
        batch.y = batch.y.squeeze().long() ### CODICE ORIGINALE
#         batch.y = batch.y.long()
        
        loss = F.cross_entropy(y_pred, batch.y)
        loss.backward()
        optimizer.step()
        print(f"=== LOSS in train() backward: {loss}")
        
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
            y_ = model(batch)

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

    plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['benign', 'malware'], yticklabels=['benign', 'malware'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

    print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(
        test_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100))

    return accuracy, precision, recall, f1

#### MAIN ####

# CPG_generator()

# Embed_generator()
 
context = configs.Process()
input_dataset = loads(PATHS.input)

# # remove samples without edges
# input_dataset = input_dataset[input_dataset['input'].apply(lambda x: x.edge_index.size(1) > 0)]
# # standardize feature vector for handle 0 values in node feature vector
# for id, data in input_dataset.input.items():
#     data.x = (data.x - data.x.mean(dim=0)) / (data.x.std(dim=0) + 1e-6)

# split the dataset and pass to DataLoader with batch size
# train_loader, val_loader, test_loader = list(
#     map(lambda x: x.get_loader(context.batch_size, shuffle=context.shuffle),
#         train_val_test_split(input_dataset, shuffle=context.shuffle)))

input_dataset = input_dataset.reset_index(drop=True)
input_dataset = InputDataset(input_dataset)
train_loader = input_dataset.get_loader(3, shuffle=context.shuffle)
# each element in codes and types is a list with the codes and types of each function graph


val_loader = None
test_loader = None

# print(f'=== run.py - DataLoader: {len(train_loader)} {len(val_loader)} {len(test_loader)} ====')

Bertggnn = configs.BertGGNN()
gated_graph_conv_args = Bertggnn.model["gated_graph_conv_args"]
conv_args = Bertggnn.model["conv_args"]
emb_size = Bertggnn.model["emb_size"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = BertGGCN(gated_graph_conv_args, conv_args, emb_size, device).to(device)
# model = BertGGCN(gated_graph_conv_args, conv_args, emb_size, device)
optimizer = torch.optim.AdamW(model.parameters(), lr=Bertggnn.learning_rate, weight_decay=Bertggnn.weight_decay)

best_acc = 0.0
NUM_EPOCHS = context.epochs
PATH = "data/model/vinz_updateNodes_model"
for epoch in range(1, NUM_EPOCHS + 1):
    train(model, device, train_loader, optimizer, epoch)
    # acc, precision, recall, f1 = validate(model, DEVICE, val_loader)
    # if best_acc <= acc:
    #     best_acc = acc
    #     torch.save(model.state_dict(), PATH)
    # print("acc is: {:.4f}, best acc is {:.4f}n".format(acc, best_acc))

# model_test = BertGGCN(gated_graph_conv_args, conv_args, emb_size, device).to(device)
# # model_test = BertGGCN(gated_graph_conv_args, conv_args, emb_size, device)
# model_test.load_state_dict(torch.load(args.path))
# accuracy, precision, recall, f1 = test(model_test, DEVICE, test_loader)


