"""
@inproceedings{cai2021time,
  title={Time series domain adaptation via sparse associative structure alignment},
  author={Cai, Ruichu and Chen, Jiawei and Li, Zijian and Chen, Wei and Zhang, Keli and Ye, Junjian and Li, Zhuozhang and Yang, Xiaoyan and Zhang, Zhenjie},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={8},
  pages={6859--6867},
  year={2021}
}
"""

from torch import nn
from transformers import PretrainedConfig, PreTrainedModel


class SASAConfigBase(PretrainedConfig):
    """Base config for sasa model"""

    n_input_channels: int
    window_size: int
    time_interval: int
    dense_dim: int
    drop_prob: float
    coeff: float

    def __init__(
        self,
        n_input_channels: int = 9,
        window_size: int = 6,
        time_interval: int = 1,
        dense_dim: int = 100,
        drop_prob: float = 0.0,
        coeff: float = 10,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_input_channels = n_input_channels
        self.window_size = window_size
        self.time_interval = time_interval
        self.dense_dim = dense_dim
        self.drop_prob = drop_prob
        self.coeff = coeff

    @property
    def segments_length(self):
        return list(range(self.time_interval, self.window_size + 1, self.time_interval))

    @property
    def segments_num(self):
        return len(self.segments_length)


class SASAConfig(SASAConfigBase):
    """Config for sasa model"""

    model_type: str = "SASA"

    h_dim: int
    lstm_layer: int

    def __init__(self, h_dim: int = 64, lstm_layer: int = 1, **args):
        super().__init__(**args)
        self.h_dim = h_dim
        self.lstm_layer = lstm_layer


class SASABackbone(PreTrainedModel):

    config_class = SASAConfig
    config: SASAConfig

    def __init__(self, config: SASAConfig):
        super(SASABackbone, self).__init__(config)

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.config.h_dim,
            num_layers=self.config.lstm_layer,
            batch_first=True,
        )

    def forward(self, x):
        x = self.lstm(x)
        return (x,)
