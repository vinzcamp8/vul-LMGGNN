import argparse
from argparse import ArgumentParser
import gc
import os
import shutil
import configs
import utils.data as data
import utils.process as process
import utils.functions.cpg_mod as cpg
import torch
import numpy as np
from models.LMGNN import BertGGCN
from baseline.training_val_test import load_checkpoint
from torch_geometric.data import Data
import time


'''
Load the configuration parameters from the configs.json file
'''
PATHS = configs.Paths()
FILES = configs.Files()
DEVICE = FILES.get_device()

def parse_CPG_single_file(file_path, label):
    """
    Parses a single file to generate its Code Property Graph (CPG) and save it as a `.pkl` file.
    
    Args:
        file_path (str): The path to the file to be processed.
    """
    context = configs.Create()

    if os.path.exists(PATHS.joern):
        shutil.rmtree(PATHS.joern)  # Clear out any remaining files in the Joern directory

    # Step 1: Generate CPG binary file for the file
    cpg_file = process.joern_parse(context.joern_cli_dir, file_path, PATHS.cpg, f"demo_{FILES.cpg}")
    print(f"File {file_path} parsed to cpg binary file demo_{FILES.cpg}.")

    # Step 2: Create CPG with graphs JSON file
    json_file = process.joern_create(context.joern_cli_dir, PATHS.cpg, PATHS.cpg, [cpg_file])[0]
    print(f"CPG binary file for {file_path} converted to JSON.")

    # Step 3: Process JSON to extract graph data and save as a `.pkl` file
    graphs = process.json_process(PATHS.cpg, json_file)
    if graphs is None:
        print(f"File {file_path} not processed due to missing or empty graphs.")
        return
    dataset = data.create_with_index(graphs, ["Index", "cpg"])
    dataset["target"] = label
    dataset["func"] = open(file_path, 'r').read()

    print(f"Writing CPG dataset for {file_path} to .pkl.")
    data.write(dataset, PATHS.cpg, f"demo_{FILES.cpg}.pkl")

    # Step 4: Clean up files
    # Delete temporary `.bin` and `.json` files after writing `.pkl` to free up space
    os.remove(PATHS.cpg+cpg_file)  # Remove .bin file
    os.remove(PATHS.cpg+json_file)  # Remove .json file
    
    print(f"CPG generation for {file_path} completed.")

    return f"demo_{FILES.cpg}.pkl"


def embed_single_pkl(pkl_file):
    """
    Generates embeddings for a single `.pkl` file containing a CPG dataset.

    Args:
        pkl_file (str): The path to the `.pkl` file to be processed.
    """
    context = configs.Embed()
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
    data.write(cpg_dataset[["input", "target", "func"]], PATHS.input, f"{file_name}_{FILES.input}")

    pyg_data = cpg_dataset.iloc[0].input
    pyg_data.func = cpg_dataset.iloc[0].func

    return f"{PATHS.input}{file_name}_{FILES.input}", pyg_data

def demo(model, device, sample, path_output_results=None):
    """
    Performs inference on a single sample using the provided model.

    :param model: The model to be used for inference.
    :param device: The device to perform inference on (e.g., 'cpu' or 'cuda').
    :param sample: The single sample to perform inference on.
    :return: Tuple containing the true label, predicted label, and predicted probability.
    """
    if model.__class__.__name__ == 'BertGGCN':
        vul_lmgnn_flag = True
    else:
        vul_lmgnn_flag = False

    model.eval()
    sample.to(device)
    with torch.no_grad():
        # vul_lmgnn need the entire sample cause use other features like "func"
        y_ = model(sample) if vul_lmgnn_flag else model(sample.x, sample.edge_index, sample.batch)

    sample.y = sample.y.squeeze().long()
    pred = y_.max(-1, keepdim=True)[1]
    y_prob = torch.softmax(y_, dim=1).cpu().numpy()[:, 1]

    true_label = sample.y.cpu().numpy()
    predicted_label = pred.cpu().numpy()
    predicted_probability = y_prob

    print('Sample - True label: {}, Predicted label: {}, Predicted probability: {:.6f}'.format(
        true_label, predicted_label, predicted_probability[0]))

    # results_array = np.column_stack((true_label, predicted_label, predicted_probability))
    # header_text = "True label, Predicted label, Predicted Probability"
    # with open(f'{path_output_results}{model.__class__.__name__}_demo_probabilities.txt', 'a') as f:
    #     np.savetxt(f, results_array, fmt='%1.6f', delimiter='\t', header=header_text)

    return true_label, predicted_label, predicted_probability

'''
directory name: vul_lmgnn_1e-05_4_20_1e-06_0.5

Means:
Training on Vul-LMGNN model with:
- learning_rate 1e-5
- batch_size 4
- epochs 20
- weight_decay 1e-06
- pred_lambda 0.5 (this hyperparametr is only for Vul-LMGNN)
'''

def parse_hyperparameters_from_foldername(foldername):
    """
    Parses the hyperparameters from the folder name in which the model is saved.

    Args:
        foldername (str): The folder name containing the hyperparameters.

    Returns:
        Tuple containing the hyperparameters for the model.
    """
    hyperparams = foldername.split("_")
    learning_rate = float(hyperparams[2])
    batch_size = int(hyperparams[3])
    epochs = int(hyperparams[4])
    weight_decay = float(hyperparams[5])
    pred_lambda = float(hyperparams[6])

    return learning_rate, batch_size, epochs, weight_decay, pred_lambda

if __name__ == '__main__':

    parser: ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Path to the model checkpoint file.')
    parser.add_argument('--sample_path', type=str, help='Path to the sample file to be processed.')
    
    args = parser.parse_args()
    
    '''
    example: python3 demo.py --model_path="data/model/vul_lmgnn_5e-05_32_10_1e-05_0.5" --sample_path="data/demo/0.c"
    '''
    start_time = time.time()
    cpg_file = parse_CPG_single_file(args.sample_path, 1)
    time_to_generate_cpg = time.time() - start_time
    print("\nTime to generate CPG: {:.2f} seconds.\n".format(time_to_generate_cpg))
    embed_file , pyg_data = embed_single_pkl(cpg_file)
    time_to_embed = time.time() - time_to_generate_cpg
    print("\nTime to embed: {:.2f} seconds.\n".format(time_to_embed))
    

    # Load the model
    args.learning_rate, args.batch_size, args.epochs, args.weight_decay, args.pred_lambda = parse_hyperparameters_from_foldername(args.model_path)
    args.patience = 5

    Bertggnn = configs.BertGGNN()

    Bertggnn.update_from_args(args)
    gated_graph_conv_args = Bertggnn.model["gated_graph_conv_args"]
    conv_args = Bertggnn.model["conv_args"]
    emb_size = Bertggnn.model["emb_size"]
    pred_lambda = Bertggnn.pred_lambda 
    model = BertGGCN(pred_lambda, gated_graph_conv_args, conv_args, emb_size, DEVICE).to(DEVICE)
    model = load_checkpoint(model, args.model_path+"/vul_lmgnn_checkpoint.pth")
    
    start_time = time.time()
    demo(model, DEVICE, pyg_data)
    time_to_inference = time.time() - start_time
    print("\nTime to inference: {:.2f} seconds.\n".format(time_to_inference))
    
