from transformers import PreTrainedModel, PretrainedConfig
from torch import nn

"""
CoDATS model as decribed in https://arxiv.org/pdf/2005.10996.pdf. 
Implementation details taken from the https://github.com/emadeldeen24/AdaTime benchmark and from the original paper.
"""


class DannConfig(PretrainedConfig):

    n_source_dom: int  # number of source domains
    class_loss_w: float  # weight of the classification loss
    disc_loss_w: float  # weight of the discriminator loss

    def __init__(
        self, n_source_dom: int = 1, class_loss_w: float = 1.0, disc_loss_w: float = 1.0, **kwargs
    ):
        super().__init__(**kwargs)
        self.n_source_dom = n_source_dom
        self.class_loss_w = class_loss_w
        self.disc_loss_w = disc_loss_w


class CoDATSConfig(DannConfig):

    model_type: str = "CoDATS"

    n_input_channels: int  # number of input channels
    avgpool_to: int  # output length for the feature extractor

    def __init__(self, n_input_channels: int = 1, avgpool_to: int = 1, **args):
        super().__init__(**args)
        self.n_input_channels = n_input_channels
        self.avgpool_to = avgpool_to


class Feature_extractor(PreTrainedModel):

    config_class = CoDATSConfig
    config: CoDATSConfig

    conv_block1: nn.Sequential
    conv_block2: nn.Sequential
    conv_block3: nn.Sequential
    avgpool: nn.AdaptiveAvgPool1d

    def __init__(self, config: CoDATSConfig):
        super().__init__(config)

        def Conv_block(
            in_channels: int, out_channels: int, kernel_size: int, stride: int
        ) -> nn.Sequential:

            conv_block = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    bias=False,
                    padding="same",
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            )

            return conv_block

        self.conv_block1 = Conv_block(self.config.n_input_channels, 128, 9, 1)

        self.conv_block2 = Conv_block(128, 256, 5, 1)

        self.conv_block3 = Conv_block(256, 128, 3, 1)

        self.avgpool = nn.AdaptiveAvgPool1d(self.config.avgpool_to)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.avgpool(x)
        x_flat = x.reshape(x.shape[0], -1)
        return (x_flat,)


class Classifier(nn.Module):

    linear: nn.Linear

    def __init__(self, avgpool_to: int, num_labels: int):
        super().__init__()
        self.linear = nn.Linear(avgpool_to * 128, num_labels)

    def forward(self, x):
        return self.linear(x)


class Discriminator(nn.Module):

    linear1: nn.Sequential
    linear2: nn.Sequential
    linear3: nn.Linear

    def __init__(self, avgpool_to: int, n_source_dom: int):
        super().__init__()

        def linear_block(in_features, out_features):
            Linear_Block = nn.Sequential(
                nn.Linear(in_features, out_features, bias=False),
                nn.BatchNorm1d(out_features),
                nn.ReLU(),
                nn.Dropout(p=0.3),
            )

            return Linear_Block

        self.linear1 = linear_block(avgpool_to * 128, 500)

        self.linear2 = linear_block(500, 500)

        self.linear3 = nn.Linear(500, n_source_dom + 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return self.linear3(x)
