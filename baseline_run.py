import argparse
from argparse import ArgumentParser
import torch

baseline_models = ['reveal', 'ivdetect', 'dummy', 'mlp', 'gcn', 'gat']

if __name__ == '__main__':
    parser: ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('-model', '--model', type=str, nargs='+', help='Baseline models to run. Options: reveal, ivdetect, dummy, mlp, gcn, gat.')
    parser.add_argument('-train', '--train', action="store_true", help='Start the training process. Specify hyperparameters.')
    parser.add_argument('-test', '--test', action="store_true", help='Start the testing process. Specify hyperparameters.')
    # Hyperparameters
    parser.add_argument('-learning_rate', '--learning_rate', type=float, nargs='+', help='Hyperparameter: List of learning rates for the model.')
    parser.add_argument('-batch_size', '--batch_size', type=int, help='Hyperparameter: Batch size for training.')
    parser.add_argument('-epochs', '--epochs', type=int, help='Hyperparameter: Number of epochs for training.')
    parser.add_argument('-weight_decay', '--weight_decay', type=float, nargs='+', help='Hyperparameter: Weight decay for the optimizer.')
    parser.add_argument('-patience', '--patience', type=int, help='Hyperparameter: Patience for early stopping.')

    '''
    python3 -u baseline_run.py -model reveal -train -learning_rate 1e-04 -batch_size 8 -epochs 10 -weight_decay 1e-04 -patience 3
    python3 -u baseline_run.py -model mlp -train -learning_rate 1e-04 -batch_size 8 -epochs 3 -weight_decay 1e-04 -patience 3
    '''
    
    args = parser.parse_args()
    print("Run with args:", args)

    if not any(model in baseline_models for model in args.model):
        print("Error: None of the specified models are in the list of baseline models.")
        exit(1)

    print("Loading data...")
    train_loader = torch.load(f"input/bs_{args.batch_size}/train_loader.pth")
    val_loader = torch.load(f"input/bs_{args.batch_size}/val_loader.pth")
    test_loader = torch.load(f"input/bs_{args.batch_size}/test_loader.pth")

    learning_rates = args.learning_rate
    weight_decays = args.weight_decay
    for model in args.model:

        if model == 'dummy':
            print("Running Dummy...")
            from baseline.run_dummy import run_dummy
            run_dummy(train_loader, val_loader, test_loader)
        for learning_rate in learning_rates:
            for weight_decay in weight_decays:
                args.learning_rate = learning_rate
                args.weight_decay = weight_decay

                if model == 'reveal':
                    print("Running ReVeal...")
                    from baseline.run_reveal import run_reveal
                    run_reveal(args, train_loader, val_loader, test_loader)

                elif model == 'ivdetect':
                    print("Running IVDetect...")
                    from baseline.run_ivdetect import run_ivdetect
                    run_ivdetect(args, train_loader, val_loader, test_loader)

                elif model == 'gcn':
                    print("Running GCN...")
                    from baseline.run_gcn import run_gcn
                    run_gcn(args, train_loader, val_loader, test_loader)
                
                elif model == 'mlp':
                    print("Running MLP...")
                    from baseline.run_mlp import run_mlp
                    run_mlp(args, train_loader, val_loader, test_loader)
                
                elif model == 'gat':
                    print("Running GAT...")
                    from baseline.run_gat import run_gat
                    run_gat(args, train_loader, val_loader, test_loader)