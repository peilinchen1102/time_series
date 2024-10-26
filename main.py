import torch
from src.utils import load_data
from src.tokenization import tokenize_time_series
from src.positional_encoding import positional_encoding
from src.transformer_model import TimeSeriesTransformer

# Hyperparameters
WINDOW_SIZE = 10
NUM_FEATURES = 2

def main() -> None:
    # Load and prepare data
    time_series = load_data("data/time_series_data.csv")
    sequence_length = len(time_series) // WINDOW_SIZE
    input_dim = WINDOW_SIZE + NUM_FEATURES  # Dimension of token embedding

    # Tokenize the data
    tokens = tokenize_time_series(time_series, WINDOW_SIZE, NUM_FEATURES)

    # Generate positional encodings
    pos_encoding = positional_encoding(sequence_length, input_dim)

    # Combine tokens with positional encodings
    model_input = tokens + pos_encoding[:tokens.shape[0], :]

    # Initialize and run the transformer model
    transformer = TimeSeriesTransformer(input_dim)
    model_output = transformer.forward(model_input.unsqueeze(0))  # Add batch dimension

    print("Model Output Shape:", model_output.shape)

if __name__ == "__main__":
    main()