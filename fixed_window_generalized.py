import torch
from src.transformer_model import TimeSeriesTransformer
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from src.dataset import PTBXL
import matplotlib.pyplot as plt
from load_datasets.catalog import DATASET_DICT
import sys
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(train_dataset, test_dataset) -> None:
    
    # TODO: preprocessing
    print("train_dataset shape", train_dataset[0][1].shape)
    seq_len, new_window_size, channels = train_dataset[0][1].shape
    

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    print("Train dataset ready")
    print("num classes", train_dataset.num_classes())
    input_dim = new_window_size * channels   # Number of features (12 channels + positional encoding) * window size
    d_model = 64                             # Dimension of embeddings (output from transformer)
    num_heads = 8                            # Number of attention heads
    num_layers = 6                           # Number of encoder layers
    dim_feedforward = 256                    # Feedforward layer dimension
    dropout = 0.1                            # Dropout rate
    num_classes = train_dataset.num_classes()
    learning_rate = 1e-4
    num_epochs = 10

    model = TimeSeriesTransformer(
        input_dim=input_dim,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_classes=num_classes,
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training started!")

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for idx, inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)
        print(f'Average Loss: {average_loss:.4f}')
        torch.cuda.empty_cache()
    print("Training complete!")

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time} seconds")

    model.eval()

    correct = 0
    total = 0

    print("Prep test dataset")
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    print("Evaluation starts!")

    with torch.no_grad():
        for idx, inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        torch.cuda.empty_cache()
    accuracy = 100 * correct / total
    print(f'Accuracy on test data: {accuracy:.2f}%')

if __name__ == "__main__":

    print("Pipeline started!")
    
    print("GPU:", torch.cuda.current_device())
    dataset_name = sys.argv[1]
    dataset = DATASET_DICT[dataset_name]
    train_dataset = dataset(base_root="../local_ecg", window_size=10, overlap=5, train=True, download=False, dataset_name=dataset_name)
    test_dataset = dataset(base_root="../local_ecg", window_size=10, overlap=5, train=False, download=False, dataset_name=dataset_name)
    
    print(f"Train dataset length: {train_dataset.subject_data.shape}")
    print(f"Test dataset length: {test_dataset.subject_data.shape}")
    main(train_dataset, test_dataset)




   