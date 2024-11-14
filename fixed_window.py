import torch
from src.utils import load_data, pad_window_interval
from src.tokenization import tokenize
from src.positional_encoding import positional_encoding
from src.transformer_model import TimeSeriesTransformer
from src.change_points import detect_changes
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from src.dataset import PTBXL
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(X_train, y_train, X_test, y_test, window_size=None, overlap=10) -> None:
    
    # TODO: preprocessing

    model_inputs = []
    for i, time_series in enumerate(X_train): # (19000, 1000, 12)
        change_points = detect_changes(time_series)
        tokens = tokenize(torch.from_numpy(time_series), window_size, overlap, change_points)
        pos_encoding = positional_encoding(tokens.shape[0], tokens.shape[1]).unsqueeze(-1)
        model_input = torch.cat((tokens, pos_encoding), dim=-1)
        model_inputs.append(model_input)
        print("Time series data: ", i)
    
    input_seq = torch.tensor(model_inputs)
    # padded_window_seq = pad_window_interval(model_inputs)
    # padded_input_seq = pad_sequence(padded_window_seq, batch_first=True)
    # padded_input_seq = padded_input_seq.float()

    dataset = PTBXL(input_seq, y_train.to(device))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    print("Dataset ready")
    size, seq_len, window_size, channels = input_seq.shape

    print("Window size", window_size)
    print("Channels", channels)

    input_dim = window_size * channels   # Number of features (12 channels + positional encoding) * window size
    d_model = 64                         # Dimension of embeddings (output from transformer)
    num_heads = 8                        # Number of attention heads
    num_layers = 6                       # Number of encoder layers
    dim_feedforward = 256                # Feedforward layer dimension
    dropout = 0.1                        # Dropout rate
    num_classes = 5
    learning_rate = 1e-4
    num_epochs = 3

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

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        average_loss = running_loss / len(dataloader)
        print(f'Average Loss: {average_loss:.4f}')
        torch.cuda.empty_cache()
    print("Training complete!")

    model.eval()

    correct = 0
    total = 0

    print("Prep test dataset")
    model_inputs = []
    for i, time_series in enumerate(X_test): # (19000, 1000, 12)
        change_points = detect_changes(time_series)
        tokens = tokenize(torch.from_numpy(time_series), window_size, overlap, change_points)
        pos_encoding = positional_encoding(tokens.shape[0], tokens.shape[1]).unsqueeze(-1)
        model_input = torch.cat((tokens, pos_encoding), dim=-1)
        model_inputs.append(model_input)
        print("Time series data: ", i)

    input_seq = torch.tensor(model_inputs)
    # padded_window_seq = pad_window_interval(model_inputs)
    # padded_input_seq = pad_sequence(padded_window_seq, batch_first=True)
    # padded_input_seq = padded_input_seq.float()

    dataset = PTBXL(input_seq, y_test.to(device))
    test_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    print("Evaluation starts!")

    with torch.no_grad():
        for inputs, labels in test_loader:
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
    # Load and prepare data
    path = './dataset/ecg/WFDB_PTBXL/ptbxl/'
    sampling_rate=100
    X_train, y_train, X_test, y_test = load_data(path, sampling_rate)

    # remove data without labels
    mask = y_train.apply(lambda x: isinstance(x, list) and len(x) > 0)
    y_train = y_train[mask][:100]
    X_train = X_train[mask][:100]
    y_train = y_train.apply(lambda x: x[0]).astype('category')
    y_train = torch.tensor(y_train.cat.codes.values, dtype=torch.long)

    mask = y_test.apply(lambda x: isinstance(x, list) and len(x) > 0)
    y_test = y_test[mask][:100]
    X_test = X_test[mask][:100]
    y_test = y_test.apply(lambda x: x[0]).astype('category')
    y_test = torch.tensor(y_test.cat.codes.values, dtype=torch.long)

    print("Data loaded")
    main(X_train, y_train, X_test, y_test, 300)


   