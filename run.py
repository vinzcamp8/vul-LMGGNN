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
from utils.data.datamanager import loads, train_val_test_split
from models.LMGNN import BertGGCN
from utils.process.training_val_test import train, validate, test, save_checkpoint, load_checkpoint
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
        data.write(cpg_dataset[["input", "target", "func"]], PATHS.input, f"{file_name}_{FILES.input}")

        del cpg_dataset
        gc.collect()

def Dataloaders_generator(args, save=False):
    context = configs.Process()
    context.update_from_args(args)
    if save:
        path = f"input/bs_{context.batch_size}/"
        os.makedirs(path)

    input_dataset = loads(PATHS.input)

    # split the dataset and pass to DataLoader with batch size
    train_loader, val_loader, test_loader = list(
        map(lambda x: x.get_loader(context.batch_size, shuffle=context.shuffle),
            train_val_test_split(input_dataset, shuffle=context.shuffle)))
    
    print(f'=== run.py - DataLoaders: {len(train_loader)} {len(val_loader)} {len(test_loader)} ====')

    if save:
        torch.save(train_loader, f"{path}/train_loader.pth")
        torch.save(val_loader, f"{path}/val_loader.pth")
        torch.save(test_loader, f"{path}/test_loader.pth")
        print(f"DataLoaders saved in {path}")

    return train_loader, val_loader, test_loader

def Training_Validation_Vul_LMGNN(args, train_loader, val_loader):
    context = configs.Process()
    context.update_from_args(args)
    Bertggnn = configs.BertGGNN()
    Bertggnn.update_from_args(args)

    gated_graph_conv_args = Bertggnn.model["gated_graph_conv_args"]
    conv_args = Bertggnn.model["conv_args"]
    emb_size = Bertggnn.model["emb_size"]
    early_stop_patience = context.patience

    learning_rate = Bertggnn.learning_rate
    batch_size = context.batch_size
    epochs = context.epochs
    weight_decay = Bertggnn.weight_decay
    pred_lambda = Bertggnn.pred_lambda

    # Initialize model, optimizer, and scheduler
    model = BertGGCN(pred_lambda, gated_graph_conv_args, conv_args, emb_size, DEVICE).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_f1 = 0.0
    early_stop_counter = 0
    path_output_model = f"{PATHS.model}vul_lmgnn_{learning_rate}_{batch_size}_{epochs}_{weight_decay}_{pred_lambda}/"
    os.makedirs(path_output_model)
    print("Starting training with args:", args)
    
    for epoch in range(1, epochs + 1):
        # Training step
        train(model, DEVICE, train_loader, optimizer, epoch)
        
        # Validation step
        acc, precision, recall, f1 = validate(model, DEVICE, val_loader, path_output_model, epoch)
        print(f"Validation - Epoch {epoch} -", "acc: {:.4f}, prec: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(acc, precision, recall, f1))

        # Save checkpoint if F1 improves
        if f1 > best_f1:
            print("New best F1 score, before was: {:.4f}\nSaving model...".format(best_f1))
            best_f1 = f1
            early_stop_counter = 0
            
            # Save model checkpoint
            checkpoint_path = str(path_output_model+"vul_lmgnn_checkpoint.pth")
            save_checkpoint(epoch, model, best_f1, checkpoint_path, optimizer, scheduler)
        else:
            early_stop_counter += 1
            print(f"No improvement in F1 score for {early_stop_counter} consecutive epochs.")

        # Update the learning rate
        scheduler.step()
        print(f"Epoch {epoch} completed. Learning rate: {scheduler.get_last_lr()[0]:.6f}")

        # Early stopping condition
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping triggered. No improvement in F1 score for {early_stop_patience} consecutive epochs.")
            break




def Testing_Vul_LMGNN(args, test_loader, model_path, model_name):
    Bertggnn = configs.BertGGNN()
    Bertggnn.update_from_args(args)
    gated_graph_conv_args = Bertggnn.model["gated_graph_conv_args"]
    conv_args = Bertggnn.model["conv_args"]
    emb_size = Bertggnn.model["emb_size"]
    pred_lambda = Bertggnn.pred_lambda 

    model_test = BertGGCN(pred_lambda, gated_graph_conv_args, conv_args, emb_size, DEVICE).to(DEVICE)
    path_checkpoint = str(model_path+model_name)
    model_test = load_checkpoint(model_test, path_checkpoint)
    accuracy, precision, recall, f1 = test(model_test, DEVICE, test_loader, model_path)
    print(f"=== Testing results: accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1} ===")


if __name__ == '__main__':
    parser: ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('-cpg', '--cpg', action="store_true", help='Specify to perform CPG generation task.')
    parser.add_argument('-embed', '--embed', action="store_true", help='Specify to perform Embedding generation task.')
    parser.add_argument('-dataloaders', '--dataloaders', default=None, help='Generate DataLoaders from input pkl files. Specify "save" or "not_save".')
    parser.add_argument('-train', '--train', action="store_true", help='Start the training process. Specify hyperparameters.')
    parser.add_argument('-test', '--test', action="store_true", help='Start the testing process. Specify hyperparameters.')
    # Hyperparameters
    parser.add_argument('-learning_rate', '--learning_rate', type=float, nargs='+', help='Hyperparameter: List of learning rates for the model.')
    parser.add_argument('-batch_size', '--batch_size', type=int, help='Hyperparameter: Batch size for training.')
    parser.add_argument('-epochs', '--epochs', type=int, help='Hyperparameter: Number of epochs for training.')
    parser.add_argument('-weight_decay', '--weight_decay', type=float, nargs='+', help='Hyperparameter: Weight decay for the optimizer.')
    parser.add_argument('-patience', '--patience', type=int, help='Hyperparameter: Patience for early stopping.')
    parser.add_argument('-pred_lambda', '--pred_lambda', type=float, nargs='+', help='Hyperparameter: Lambda for interpolating predictions. λ = 1 signifies use only Vul-LMGNN, λ = 0 use only CodeBERT.')
    
    args = parser.parse_args()
    print("Run with args:", args)
    
    '''
    Example of running the script:
    python3 run.py -cpg -embed -dataloaders save -train -test -learning_rate 0.0001 -batch_size 32 -epochs 10 -weight_decay 0.00001 -patience 5
    python3 run.py -cpg -embed 
    python3 run.py -dataloaders save -batch_size 32
    python3 run.py -train -test -learning_rate 5e-5 -batch_size 32 -epochs 3 -weight_decay 1e-6 -pred_lambda 0.5
    nohup python3 run.py -train -test -learning_rate 5e-5 -batch_size 32 -epochs 3 -weight_decay 1e-6 -pred_lambda 0.5 >> train.log 2>&1 &
    '''

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
        print("Generating DataLoader objects...")
        if args.dataloaders == "save":
            train_loader, val_loader, test_loader = Dataloaders_generator(args, save=True)
        elif args.dataloaders == "not_save":
            train_loader, val_loader, test_loader = Dataloaders_generator(args, save=False)
        else:
            exit("Specify 'save' or 'not_save' to save or not the DataLoader objects.")
    ###

    '''
    Training_Validation_Vul_LMGNN(), train and validate the model
    Input: DataLoader objects
    Output: trained model
    Parameter configs.json: 
    '''
    ###
    if args.train:
        if args.learning_rate and args.weight_decay and args.pred_lambda:
            learning_rates = args.learning_rate
            weight_decays = args.weight_decay
            pred_lambdas = args.pred_lambda
            for lr in learning_rates:
                args.learning_rate = lr
                for wd in weight_decays:
                    args.weight_decay = wd
                    for pl in pred_lambdas:
                        args.pred_lambda = pl
                        print(args)
                        if not 'train_loader' in locals() or not 'val_loader' in locals():
                            print("Loading DataLoader objects...")
                            train_loader = torch.load(f"input/bs_{args.batch_size}/train_loader.pth")
                            val_loader = torch.load(f"input/bs_{args.batch_size}/val_loader.pth")
                            print("DataLoader objects loaded.")
                        Training_Validation_Vul_LMGNN(args, train_loader, val_loader)
    ### 

    '''
    Testing_Vul_LMGNN(), test the model
    Input: DataLoader objects
    Output: test results
    Parameter configs.json: 
    '''
    ###
    args = parser.parse_args() # reload args (due to previous modification in training)
    if args.test:
        if args.learning_rate and args.weight_decay and args.pred_lambda:
            learning_rates = args.learning_rate
            weight_decays = args.weight_decay
            pred_lambdas = args.pred_lambda
            for lr in learning_rates:
                args.learning_rate = lr
                for wd in weight_decays:
                    args.weight_decay = wd
                    for pl in pred_lambdas:
                        args.pred_lambda = pl
                        if not 'test_loader' in locals() or not 'model' in locals():
                            print("Loading TestLoader...")
                            test_loader = torch.load(f"input/bs_{args.batch_size}/test_loader.pth")
                        model_path = f"{PATHS.model}vul_lmgnn_{args.learning_rate}_{args.batch_size}_{args.epochs}_{args.weight_decay}_{args.pred_lambda}/"
                        model_name = "vul_lmgnn_checkpoint.pth"
                        print("Starting Test of Model:", model_path+model_name)
                        Testing_Vul_LMGNN(args, test_loader, model_path, model_name)
    ##

