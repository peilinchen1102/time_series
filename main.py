import torch
from src.utils import load_data, pad_window_interval
from src.tokenization import tokenize
from src.positional_encoding import positional_encoding
from src.transformer_model import TimeSeriesTransformer
from src.change_points import detect_changes
from torch.nn.utils.rnn import pad_sequence


def main(window_size=None, overlap=10) -> None:
    # Load and prepare data
    path = './dataset/ecg/WFDB_PTBXL/ptbxl/'
    sampling_rate=100
    X_train, y_train, X_test, y_test = load_data(path, sampling_rate)
    
    # TODO: preprocessing

    model_inputs = []

    # (19000, 1000, 12)
    for time_series in X_train[0:5]:
        change_points = detect_changes(time_series)
        # if window_size:
        #     sequence_length = time_series.shape[0] // window_size
        # else:
        #     sequence_length = len(change_points) + 1

        # Tokenize the data
        tokens = tokenize(torch.from_numpy(time_series), window_size, overlap, change_points)

        # Generate positional encodings
        pos_encoding = positional_encoding(tokens.shape[0], tokens.shape[1]).unsqueeze(-1) # (4, 442, 1)
        
        model_input = torch.cat((tokens, pos_encoding), dim=-1)
    
        model_inputs.append(model_input)

   
    padded_window_seq = pad_window_interval(model_inputs)
    padded_input_seq = pad_sequence(padded_window_seq, batch_first=True)

    print(padded_input_seq.shape)
    #     print(model_input)
    #     break
    # model_inputs = torch.tensor(model_inputs, dtype=torch.float32)


    # # Initialize and run the transformer model
    # transformer = TimeSeriesTransformer(input_dim)
    # model_output = transformer.forward(model_input.unsqueeze(0))  # Add batch dimension

    # print("Model Output Shape:", model_output.shape)

if __name__ == "__main__":
    main()