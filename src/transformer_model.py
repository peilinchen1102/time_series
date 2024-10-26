import torch
from transformers import BertModel, BertConfig
from typing import Any

class TimeSeriesTransformer:
    def __init__(self, input_dim: int) -> None:
        config = BertConfig(vocab_size=1, hidden_size=input_dim, num_hidden_layers=2)
        self.model = BertModel(config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.model(inputs_embeds=inputs)
        return outputs.last_hidden_state