from transformers import PreTrainedModel, PretrainedConfig
import torch


class MLPConfig(PretrainedConfig):

    input_len: int
    n_input_channels: int
    n_hidden: int
    out_dim: int

    model_type: str = "MLP"

    def __init__(
        self,
        input_len: int = 128,
        n_input_channels: int = 3,
        n_hidden: int = 2,
        out_dim: int = 2,
        **args,
    ):
        super().__init__(**args)
        self.input_len = input_len
        self.n_input_channels = n_input_channels
        self.n_hidden = n_hidden
        self.out_dim = out_dim


class MLPModel(PreTrainedModel):
    config_class = MLPConfig
    config: MLPConfig

    input_layer: torch.nn.Module
    hidden_layer: torch.nn.Module
    output_layer: torch.nn.Module
    input_dim: int

    def __init__(self, config: MLPConfig):
        super().__init__(config)

        self.config = config

        self.input_dim = self.config.input_len * self.config.n_input_channels

        self.input_layer = torch.reshape
        self.hidden_layer = torch.nn.Linear(self.input_dim, self.config.n_hidden)
        self.output_layer = torch.nn.Linear(self.config.n_hidden, self.config.out_dim)

    def forward(self, mts):
        x = self.input_layer(mts, (-1, self.input_dim))
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return (x,)
