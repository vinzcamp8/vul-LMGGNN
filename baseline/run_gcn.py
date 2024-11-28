import torch
import configs
import os
from utils.process.training_val_test import train, validate, test, save_checkpoint, load_checkpoint
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

PATHS = configs.Paths()

# Define the Graph Convolutional Network (GCN) model
import torch.nn as nn

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=1)

def run_gcn(args, train_loader, val_loader, test_loader):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs
    weight_decay = args.weight_decay
    early_stop_patience = args.patience

    # Initialize the model, optimizer, and loss function
    input_dim = 769
    hidden_dim = 64
    output_dim = 2

    model = GCN(input_dim, hidden_dim, output_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Train the model
    best_f1 = 0.0
    best_recall = 0.0
    early_stop_counter = 0
    path_output_model = f"{PATHS.model}gcn_{learning_rate}_{batch_size}_{epochs}_{weight_decay}/"
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
            checkpoint_path = str(path_output_model+"gcn_checkpoint.pth")
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
    model_test = GCN(input_dim, hidden_dim, output_dim).to(DEVICE)
    path_checkpoint = str(path_output_model+"gcn_checkpoint.pth")
    model_test = load_checkpoint(model_test, path_checkpoint)
    accuracy, precision, recall, f1 = test(model_test, DEVICE, test_loader, path_output_model)
    print(f"=== Testing results: accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1} ===")