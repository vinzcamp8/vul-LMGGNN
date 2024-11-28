import torch
import configs
import os
from baseline.training_val_test import train, validate, test, save_checkpoint, load_checkpoint

PATHS = configs.Paths()

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
    
def run_mlp(args, train_loader, val_loader, test_loader):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs
    weight_decay = args.weight_decay
    early_stop_patience = args.patience

    # Initialize the model and optimizer
    # Set the input dimensions for the MLP
    INPUT_DIM = train_loader.dataset[0].x.size(1)  # Feature vector size
    HIDDEN_DIM = 128  # Hidden layer size
    NUM_CLASSES = len(torch.unique(torch.cat([data.y for data in train_loader.dataset])))

    # Initialize the model, optimizer, and loss function
    model = MLP(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Train the model
    best_f1 = 0.0
    best_recall = 0.0
    early_stop_counter = 0
    path_output_model = f"{PATHS.model}mlp_{learning_rate}_{batch_size}_{epochs}_{weight_decay}/"
    os.makedirs(path_output_model)
    print("Starting training with args:", args)
    
    for epoch in range(1, epochs + 1):
        # Training step
        train(model, DEVICE, train_loader, optimizer, epoch)
        
        # Validation step
        acc, precision, recall, f1 = validate(model, DEVICE, val_loader, path_output_model, epoch)
        print(f"Validation - Epoch {epoch} -", "acc: {:.4f}, prec: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(acc, precision, recall, f1))

        # Save checkpoint if F1 improves
        if f1 > best_f1 or (f1 == best_f1 and recall > best_recall):
            print("New best F1 score, before was: {:.4f}\nSaving model...".format(best_f1))
            best_f1 = f1
            best_recall = recall
            early_stop_counter = 0
            
            # Save model checkpoint
            checkpoint_path = str(path_output_model+"mlp_checkpoint.pth")
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
    
    # Test the model
    model_test = MLP(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES).to(DEVICE)
    path_checkpoint = str(path_output_model+"mlp_checkpoint.pth")
    if not os.path.exists(path_checkpoint):
        print(f"Checkpoint file not found at {path_checkpoint}. Skipping test.")
        return
    model_test = load_checkpoint(model_test, path_checkpoint)
    accuracy, precision, recall, f1 = test(model_test, DEVICE, test_loader, path_output_model)
    print(f"=== Testing results: accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1} ===")
