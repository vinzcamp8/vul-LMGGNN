import argparse
import gc
import shutil
from argparse import ArgumentParser
import configs
import utils.data as data
import utils.process as process
import utils.functions.cpg_mod as cpg
import torch
import pandas as pd
import torch.nn.functional as F
from utils.data.datamanager import loads, train_val_test_split
from models.LMGNN import BertGGCN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from test import *
from utils.data.check_bin_json import find_bin_without_json, find_json
import os

'''
Load the configuration parameters from the configs.json file
'''
PATHS = configs.Paths()
FILES = configs.Files()
DEVICE = FILES.get_device()

def filter_select(dataset):
    non_vulnerable = dataset.loc[dataset.target == 0]
    vulnerable = dataset.loc[dataset.target == 1]

    print(f"Non-vuln: {len(non_vulnerable)}, Vuln: {len(vulnerable)}")

    vulnerable = vulnerable.loc[vulnerable.func.str.len() < 3000]

    non_vulnerable = non_vulnerable.loc[non_vulnerable.func.str.len() < 1000]
    non_vulnerable = non_vulnerable.loc[non_vulnerable.func.str.len() > 200]

    print(f"After filter on length\nNon-vuln: {len(non_vulnerable)}, Vuln: {len(vulnerable)}")

    dataset = pd.concat([non_vulnerable, vulnerable])

    # dataset = dataset.head(100)
    return dataset

def Filter_raw_dataset():
    raw = data.read(PATHS.raw, FILES.raw)

    # Here, taking the Devign dataset as an example,
    # specific modifications need to be made according to different dataset formats.
    print(f"=== Raw dataset size: {len(raw)} ===")
    filtered = data.apply_filter(raw, filter_select) # see "select" function above
    print(f"=== Filtered dataset size: {len(filtered)} ===")
    filtered = data.clean(filtered) # remove duplicates 
    print(f"=== Filtered and cleaned dataset size: {len(filtered)} ===")
    data.drop(filtered, ["hash", "project", "cwe"]) # Hash column name "hash" or "commit_id" depends on the dataset

    return filtered

# def CPG_generator(filtered_dataset):
#     """
#     Generates Code Property Graph (CPG) datasets from raw data.

#     :return: None
#     """
#     context = configs.Create()

#     slices = data.slice_frame(filtered_dataset, context.slice_size)
#     slices = [(s, slice.apply(lambda x: x)) for s, slice in slices] # it create a list of tuples (index, dataframe)

#     cpg_files = []
#     # Create CPG binary files
#     for s, slice in slices:
#         # s: index of the slice, slice: dataframe
#         data.to_files(slice, PATHS.joern)
#         cpg_file = process.joern_parse(context.joern_cli_dir, PATHS.joern, PATHS.cpg, f"{s}_{FILES.cpg}")
#         cpg_files.append(cpg_file)
#         print(f"Dataset {s} to cpg.")
#         shutil.rmtree(PATHS.joern)

#     # # Load CPG binary files               (used for memory issues)
#     # cpg_files = find_bin_without_json(PATHS.cpg)

#     # Create CPG with graphs json files
#     json_files = process.joern_create(context.joern_cli_dir, PATHS.cpg, PATHS.cpg, cpg_files)
    
#     # # Load CPG json files                 (used for memory issues)
#     # json_files = find_json(PATHS.cpg)
#     # print(len(slices), len(json_files))
#     # slices = slices[-345:]
#     # print(len(slices))

#     # Clean json and create CPG datasets
#     for (s, slice), json_file in zip(slices, json_files):
#         graphs = process.json_process(PATHS.cpg, json_file)
#         if graphs is None:
#             print(f"Dataset chunk {s} not processed.")
#             continue
#         dataset = data.create_with_index(graphs, ["Index", "cpg"])
#         dataset = data.inner_join_by_index(slice, dataset)
#         print(f"Writing cpg dataset chunk {s}.")
#         data.write(dataset, PATHS.cpg, f"{s}_{FILES.cpg}.pkl")
#         del dataset
#         gc.collect()

def CPG_generator(filtered_dataset):
    """
    Generates Code Property Graph (CPG) datasets from raw data, processing one slice at a time.
    
    Each slice's `.bin` and `.json` files are deleted after the `.pkl` dataset is created.
    """
    context = configs.Create()
    slices = data.slice_frame(filtered_dataset, context.slice_size)
    slices = [(s, slice.apply(lambda x: x)) for s, slice in slices] # it create a list of tuples (index, dataframe)
    if os.path.exists(PATHS.joern):
        shutil.rmtree(PATHS.joern)  # Clear out any remaining files in the Joern directory
    slices = slices[8:]
    # Process each slice individually
    for s, slice in slices:
        # Step 1: Generate CPG binary file for the current slice
        data.to_files(slice, PATHS.joern)
        cpg_file = process.joern_parse(context.joern_cli_dir, PATHS.joern, PATHS.cpg, f"{s}_{FILES.cpg}")
        print(f"Dataset {s} parsed to cpg binary file.")

        # Step 2: Create CPG with graphs JSON file
        json_file = process.joern_create(context.joern_cli_dir, PATHS.cpg, PATHS.cpg, [cpg_file])[0]
        print(f"CPG binary file for dataset {s} converted to JSON.")
        exit()
        # Step 3: Process JSON to extract graph data and save as a `.pkl` file
        graphs = process.json_process(PATHS.cpg, json_file)
        if graphs is None:
            print(f"Dataset chunk {s} not processed due to missing or empty graphs.")
            continue
        dataset = data.create_with_index(graphs, ["Index", "cpg"])
        dataset = data.inner_join_by_index(slice, dataset)
        
        print(f"Writing CPG dataset chunk {s} to .pkl.")
        data.write(dataset, PATHS.cpg, f"{s}_{FILES.cpg}.pkl")

        # Step 4: Clean up files
        # Delete temporary `.bin` and `.json` files after writing `.pkl` to free up space
        os.remove(PATHS.cpg+cpg_file)  # Remove .bin file
        os.remove(PATHS.cpg+json_file)  # Remove .json file
        shutil.rmtree(PATHS.joern)  # Clear out any remaining files in the Joern directory

        # Free up memory
        del dataset, graphs
        gc.collect()
        
    print("CPG generation completed.")


def CPG_generator_json_error(s, slice):
    """
    Generates Code Property Graph (CPG) datasets from raw data, processing one slice at a time.
    
    Each slice's `.bin` and `.json` files are deleted after the `.pkl` dataset is created.
    """
    context = configs.Create()
    if os.path.exists(PATHS.joern):
        shutil.rmtree(PATHS.joern)  # Clear out any remaining files in the Joern directory
    # Process each slice individually
    # Step 1: Generate CPG binary file for the current slice
    data.to_files(slice, PATHS.joern)
    cpg_file = process.joern_parse(context.joern_cli_dir, PATHS.joern, PATHS.cpg, f"{s}_{FILES.cpg}")
    print(f"Dataset {s} parsed to cpg binary file.")

    # Step 2: Create CPG with graphs JSON file
    json_file = process.joern_create(context.joern_cli_dir, PATHS.cpg, PATHS.cpg, [cpg_file])[0]
    print(f"CPG binary file for dataset {s} converted to JSON.")

    # Step 3: Process JSON to extract graph data and save as a `.pkl` file
    graphs = process.json_process(PATHS.cpg, json_file)

    # Step 4: Clean up files
    # Delete temporary `.bin` and `.json` files after writing `.pkl` to free up space
    os.remove(PATHS.cpg+cpg_file)  # Remove .bin file
    shutil.rmtree(PATHS.joern)  # Clear out any remaining files in the Joern directory

    return graphs


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
        print(f"=== Processing input dataset {file_name} with size {len(cpg_dataset)}. ===")

        tokens_dataset = data.tokenize(cpg_dataset)                             
        data.write(tokens_dataset, PATHS.tokens, f"{file_name}_{FILES.tokens}")

        cpg_dataset["nodes"] = cpg_dataset.apply(lambda row: cpg.parse_to_nodes(row.cpg, context.nodes_dim), axis=1)
        cpg_dataset["input"] = cpg_dataset.apply(lambda row: process.nodes_to_input(row.nodes, row.target, context.nodes_dim,
                                                                            context.edge_type), axis=1)
        # Filter out rows where 'input' is None
        cpg_dataset = cpg_dataset[cpg_dataset["input"].notnull()]
        
        data.drop(cpg_dataset, ["nodes"])
        print(f"=== Saving input dataset {file_name} with size {len(cpg_dataset)}. ===")
        # write(cpg_dataset[["input", "target"]], PATHS.input, f"{file_name}_{FILES.input}")
        # write(cpg_dataset[["input", "target","func"]], PATHS.input, f"{file_name}_{FILES.input}")
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
    print('Validation Confusion Matrix:')
    print(cm)


    print('Validation set - Average loss: {:.4f}, Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(
        test_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100))

    return accuracy, precision, recall, f1

def Dataloaders(save=False):
    context = configs.Process()
    if save:
        path = f"input/bs_{context.batch_size}/"
        os.makedirs(path)

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
    
    print(f'=== run.py - DataLoaders: {len(train_loader)} {len(val_loader)} {len(test_loader)} ====')

    if save:
        torch.save(train_loader, f"{path}/train_loader.pth")
        torch.save(val_loader, f"{path}/val_loader.pth")
        torch.save(test_loader, f"{path}/test_loader.pth")

    return train_loader, val_loader, test_loader

def Training_Validation_Vul_LMGNN(train_loader, val_loader, path_output_model):
    context = configs.Process()
    Bertggnn = configs.BertGGNN()
    gated_graph_conv_args = Bertggnn.model["gated_graph_conv_args"]
    conv_args = Bertggnn.model["conv_args"]
    emb_size = Bertggnn.model["emb_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BertGGCN(gated_graph_conv_args, conv_args, emb_size, device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Bertggnn.learning_rate, weight_decay=Bertggnn.weight_decay)

    best_f1 = 0.0
    NUM_EPOCHS = context.epochs
    PATH = path_output_model
    for epoch in range(1, NUM_EPOCHS + 1):
        
        train(model, device, train_loader, optimizer, epoch)
        acc, precision, recall, f1 = validate(model, DEVICE, val_loader)
        print(f"Validation - Epoch {epoch} -", "acc: {:.4f}, prec: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(acc, precision, recall, f1))

        if f1 > 0 and best_f1 <= f1:
            print("New best f1 score, before was: {:.4f}\nSaving model...".format(best_f1))
            torch.save(model.state_dict(), str(PATH))
            best_f1 = f1


def Testing_Vul_LMGNN(test_loader, model_path):
    Bertggnn = configs.BertGGNN()
    gated_graph_conv_args = Bertggnn.model["gated_graph_conv_args"]
    conv_args = Bertggnn.model["conv_args"]
    emb_size = Bertggnn.model["emb_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_test = BertGGCN(gated_graph_conv_args, conv_args, emb_size, device).to(device)
    model_test.load_state_dict(torch.load(model_path))
    accuracy, precision, recall, f1 = test(model_test, DEVICE, test_loader)
    print(f"=== Testing results: accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1} ===")


if __name__ == '__main__':
    parser: ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('-cpg', '--cpg', action="store_true", help='Specify to perform CPG generation task.')
    parser.add_argument('-embed', '--embed', action="store_true", help='Specify to perform Embedding generation task.')
    parser.add_argument('-dataloaders', '--dataloaders', default=None, help='Generate DataLoaders from input pkl files. Specify "save" or "not_save".')
    parser.add_argument('-train', '--train', default=False, help='Start the training process. Specify the Dataloaders "bs_X" to load "True" if just generated.')
    parser.add_argument('-test', '--test', default=False, help='Start the testing process. Specify the Dataloaders "bs_X" to load or "True" if just generated.')
    parser.add_argument('-path', '--path', default="data/model/Vul_LMGNN_model.pth", help='Specify the path to save or load the model.')
    args = parser.parse_args()

    '''
    Filter_raw_dataset(), filter raw dataset
    Input: raw data
    Output: filtered dataset
    Parameter configs.json: 
    '''
    ###
    if args.cpg:
        filtered_dataset = Filter_raw_dataset()
    ###

    '''
    CPG_generator(), generate CPG datasets using Joern
    Input: filtered dataset
    Output: sliced dataset in .pkl files containing CPGs [target, func, Index, cpg] (also .json files containing graphs and .bin files of Joern)
    Parameter configs.json:
    Note: Joern can print a warning "WARN MemberAccessLinker: Could not find type member." It's normal for code where the type of variable are custom types.
    '''
    ###
    if args.cpg:
        CPG_generator(filtered_dataset)
    ###

    '''
    Embed_generator(), generate embeddings from CPG datasets using BERT
    Input: CPG datasets (.pkl)
    Output: .pkl files containing embeddings
    Parameter configs.json: 
    '''
    ###
    if args.embed:
        Embed_generator()
    ### 

    '''
    Load_input_dataset(), load input dataset from .pkl files
    Input: .pkl files containing embeddings
    Output: DataLoader objects
    Parameter configs.json: 
    '''
    ###
    if args.dataloaders:
        if args.dataloaders == "save":
            train_loader, val_loader, test_loader = Dataloaders(save=True)
        elif args.dataloaders == "not_save":
            train_loader, val_loader, test_loader = Dataloaders(save=False)
    ###

    '''
    Training_Validation_Vul_LMGNN(), train and validate the model
    Input: DataLoader objects
    Output: trained model
    Parameter configs.json: 
    '''
    ###
    if args.train:
        if not args.dataloaders:
            train_loader = torch.load(f"input/{args.train}/train_loader.pth")
            val_loader = torch.load(f"input/{args.train}/val_loader.pth")
        path_output_model = args.path
        Training_Validation_Vul_LMGNN(train_loader, val_loader, path_output_model)
    ### 

    '''
    Testing_Vul_LMGNN(), test the model
    Input: DataLoader objects
    Output: test results
    Parameter configs.json: 
    '''
    ###
    if args.test:
        model_path = args.path
        if not args.dataloaders:
            torch.load(f"input/{args.test}/test_loader.pth")
        Testing_Vul_LMGNN(test_loader, model_path)
    ###