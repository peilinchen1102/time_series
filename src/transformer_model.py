import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, dim_feedforward, dropout, num_classes):
        super(TimeSeriesTransformer, self).__init__()
        
        self.input_dim = input_dim  # 13 = 12 channels + 1 positional encoding
        self.d_model = d_model
        self.input_embedding = nn.Linear(input_dim, self.d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True),
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)  # Apply LayerNorm to the final encoder output
        )
        self.batch_first = True  # Expects (batch_size, seq_len)
        self.classification_layer = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: shape [batch_size, seq_len, window_size, channels]
        batch_size, seq_len, window_size, channels = x.shape
        flatten_x = x.view(batch_size, seq_len, window_size * channels)

        # window_size * channels
        x = self.input_embedding(flatten_x)
        output = self.transformer_encoder(x)  # With batch_first=True, shape: [batch_size, seq_len, d_model]
        pooled_output = output.mean(dim=1)  # (batch_size, d_model)
        x = self.classification_layer(pooled_output) # [batch_size, num_classes]
        return x


