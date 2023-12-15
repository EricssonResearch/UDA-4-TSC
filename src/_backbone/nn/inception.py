from transformers import PreTrainedModel
import torch
from torch import nn
from _backbone.nn.cotmix import MixConfig
from _backbone.nn.codats import DannConfig
from _backbone.nn.raincoat import RaincoatConfig
from _backbone.nn.cdan import CDANConfig
from _backbone.nn.sasa import SASAConfigBase

# Inspired by https://github.com/hfawaz/InceptionTime
# Inspired by https://github.com/timeseriesAI/tsai/blob/main/tsai/models/InceptionTime.py


class InceptionConfig(MixConfig, DannConfig, RaincoatConfig, CDANConfig, SASAConfigBase):

    n_input_channels: int
    use_residual: bool
    use_bottleneck: bool
    nb_filters: int
    depth: int
    kernel_size: int
    stride: int
    bottleneck_size: int
    num_kernels: int
    block_size: int
    num_pool: int
    out_dim: int = None

    model_type: str = "Inception"

    def __init__(
        self,
        n_input_channels: int = 3,
        use_residual: bool = True,
        use_bottleneck: bool = True,
        nb_filters: int = 32,
        depth: int = 6,
        kernel_size: int = 40,
        stride: int = 1,
        bottleneck_size: int = 32,
        num_kernels: int = 3,
        block_size: int = 3,
        num_pool: int = 1,
        **args,
    ):
        super().__init__(**args)
        self.n_input_channels = n_input_channels
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.nb_filters = nb_filters
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.bottleneck_size = bottleneck_size
        self.num_kernels = num_kernels
        self.block_size = block_size
        self.num_pool = num_pool
        self.out_dim = self.nb_filters * (self.num_kernels + self.num_pool)

    @property
    def h_dim(self) -> int:
        return self.out_dim


class InceptionModule(torch.nn.Module):

    bottleneck: torch.nn.Module
    convs: torch.nn.ModuleList
    maxconvpool: torch.nn.Sequential
    bn: torch.nn.BatchNorm1d
    act: torch.nn.ReLU

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_kernels: int,
        kernel_size: int,
        use_bottleneck: bool,
        bottleneck_size: int,
        num_pool: int,
        stride: int,
    ):
        super().__init__()
        # create the kernels for each depth of the inception module
        kernel_sizes = [kernel_size // (2**i) for i in range(num_kernels)]
        # make sure they are odd not even
        kernel_sizes = [
            kernel_size if kernel_size % 2 != 0 else kernel_size + 1 for kernel_size in kernel_sizes
        ]
        # create the bottleneck
        self.bottleneck = (
            torch.nn.Conv1d(in_channels, bottleneck_size, kernel_size=1, bias=False, padding="same")
            if use_bottleneck
            else torch.nn.Identity()
        )
        # create the list of 1d convs that will be applied in parallel
        self.convs = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    bottleneck_size if use_bottleneck else in_channels,
                    out_channels,
                    kernel_size,
                    bias=False,
                    padding="same",
                )
                for kernel_size in kernel_sizes
            ]
        )
        # create the additional maxpool1d in parallel
        assert num_pool == 1, "for now we support only num_pool=1"
        self.maxconvpool = torch.nn.Sequential(
            *[
                torch.nn.MaxPool1d(kernel_size=3, stride=stride, padding=1),
                torch.nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, bias=False, padding="same"
                ),
            ]
        )
        self.bn = torch.nn.BatchNorm1d(num_features=out_channels * (num_kernels + num_pool))
        self.act = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_tensor = x
        x = self.bottleneck(input_tensor)
        x = torch.cat([conv(x) for conv in self.convs] + [self.maxconvpool(input_tensor)], dim=1)
        return self.act(self.bn(x))


# Backbone common to traditional supervised learning and to domain adaptation
# via adversarial learning
class InceptionModel(PreTrainedModel):

    config_class = InceptionConfig
    config: InceptionConfig

    inception_modules: torch.nn.ModuleList
    shortcut_modules: torch.nn.ModuleList
    act: torch.nn.ReLU
    gap: torch.nn.Sequential
    out_dim: int

    def check_if_time_use_residual(self, d: int):
        """
        checks if at depth d we need a residual connection
        """
        return self.config.use_residual and (
            (d % self.config.block_size) == (self.config.block_size - 1)
        )

    def __init__(self, config: InceptionConfig):
        super().__init__(config)

        in_channels = self.config.n_input_channels
        out_channels = self.config.nb_filters

        self.inception_modules = torch.nn.ModuleList()
        self.shortcut_modules = torch.nn.ModuleList()

        # loop through depth
        for d in range(self.config.depth):
            # add the inception module
            self.inception_modules.append(
                InceptionModule(
                    in_channels=in_channels if d == 0 else self.config.out_dim,
                    out_channels=out_channels,
                    num_kernels=self.config.num_kernels,
                    kernel_size=self.config.kernel_size,
                    use_bottleneck=self.config.use_bottleneck if d > 0 else False,
                    bottleneck_size=self.config.bottleneck_size,
                    num_pool=self.config.num_pool,
                    stride=self.config.stride,
                )
            )
            # add shortcut if needed and reached block size
            if self.check_if_time_use_residual(d):
                # set the output of shortcut
                shortcut_out_channels = self.config.out_dim
                # set the input of shortcut
                shortcut_in_channels = (
                    in_channels if d == self.config.block_size - 1 else shortcut_out_channels
                )
                # create the shortcut
                self.shortcut_modules.append(
                    torch.nn.Sequential(
                        *[
                            torch.nn.Conv1d(
                                shortcut_in_channels,
                                shortcut_out_channels,
                                kernel_size=1,
                                bias=False,
                                padding="same",
                            ),
                            torch.nn.BatchNorm1d(num_features=shortcut_out_channels),
                        ]
                    )
                )
            self.act = torch.nn.ReLU()
            self.gap = torch.nn.Sequential(
                *[torch.nn.AdaptiveAvgPool1d(output_size=1), torch.nn.Flatten()]
            )

    def forward(self, x: torch.Tensor):
        res = x
        for d in range(self.config.depth):
            x = self.inception_modules[d](x)
            if self.check_if_time_use_residual(d):
                res = x = self.act(
                    torch.add(x, self.shortcut_modules[d // self.config.block_size](res))
                )
        x = self.gap(x)
        return (x,)


# The following models are used only when training with the Adversarial Learning
# framework proposed for DANN (https://arxiv.org/pdf/1505.07818.pdf)

# Classifier for the original classification task in the Adversarial Learning framework
class Classifier(nn.Module):

    linear: nn.Linear

    def __init__(self, avgpool_to: int, num_labels: int):
        super().__init__()
        self.linear = nn.Linear(avgpool_to, num_labels)

    def forward(self, x):
        return self.linear(x)


# Discriminator to distinguish samples coming for different domains
# used in the Adversarial Learning framework
class Discriminator(nn.Module):

    linear1: nn.Sequential
    linear2: nn.Sequential
    linear3: nn.Linear

    def __init__(self, avgpool_to: int, n_source_dom: int):
        super().__init__()

        def linear_block(in_features, out_features):
            Linear_Block = nn.Sequential(
                nn.Linear(in_features, out_features), nn.ReLU(), nn.Dropout(p=0.3)
            )

            return Linear_Block

        self.linear1 = linear_block(avgpool_to, 500)

        self.linear2 = linear_block(500, 500)

        self.linear3 = nn.Linear(500, n_source_dom + 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return self.linear3(x)
