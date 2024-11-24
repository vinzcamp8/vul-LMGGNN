import torch
import configs
import os
from utils.process.training_val_test import train, validate, test
from models.reveal import Reveal

PATHS = configs.Paths()

def run_reveal(args, train_loader, val_loader, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs
    weight_decay = args.weight_decay

    # Initialize the model and optimizer
    model = Reveal().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Train the model
    best_f1 = 0.0
    best_recall = 0.0
    path_output_model = f"{PATHS.model}reveal_{learning_rate}_{batch_size}_{epochs}_{weight_decay}/"
    os.makedirs(path_output_model)
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        acc, precision, recall, f1 = validate(model, device, val_loader)
        print(f"Validation - Epoch {epoch} -", "acc: {:.4f}, prec: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(acc, precision, recall, f1))

        if f1 > 0 and best_f1 <= f1:
            print("New best f1 score, before was: {:.4f}\nSaving model...".format(best_f1))
            torch.save(model.state_dict(), str(path_output_model+"reveal_f1.pth"))
            best_f1 = f1
        
        if recall > 0 and best_recall <= recall:
            print("New best recall score, before was: {:.4f}\nSaving model...".format(best_recall))
            torch.save(model.state_dict(), str(path_output_model+"reveal_recall.pth"))
            best_recall = recall
    
    # Test the model
    model_test = Reveal().to(device)
    model_test.load_state_dict(torch.load(path_output_model+"reveal_f1.pth"))
    accuracy, precision, recall, f1 = test(model_test, device, test_loader)
    print(f"=== f1 model - Testing results: accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1} ===")

    model_test = Reveal().to(device)
    model_test.load_state_dict(torch.load(path_output_model+"reveal_recall.pth"))
    accuracy, precision, recall, f1 = test(model_test, device, test_loader)
    print(f"=== recall model - Testing results: accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1} ===")

