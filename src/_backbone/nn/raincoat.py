import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
import torch.nn.functional as F

"""Raincoat model for time series classification as described in 'Domain Adaptation for Time Series Under Feature and Label Shifts'

@inproceedings{he2023domain,
title = {Domain Adaptation for Time Series Under Feature and Label Shifts},
author = {He, Huan and Queen, Owen and Koker, Teddy and Cuevas, Consuelo and Tsiligkaridis, Theodoros and Zitnik, Marinka},
booktitle = {https://arxiv.org/abs/2302.03133},
year      = {2023}
}

Code taken from the official [implementation](https://github.com/mims-harvard/Raincoat/tree/91db75f41d74e0f18c3f6ef50f884a7fbb78b60f)
"""


class RaincoatConfigBase(PretrainedConfig):
    """Base Config for RainCoat based models"""

    sink_loss_w: float  # weight for the sinkhorn loss measuring the similarity between the source and target domain latent representations
    class_loss_w: float  # weight for the classification loss
    cont_loss_w: float  # weight for the contrastive loss
    pc_fourier_modes: float  # percentage of the time series length to take as fourier modes (number of terms to take in the series) for Spectral Conv NN

    def __init__(
        self,
        pc_fourier_modes: float = 1.0,  # percentage of fourier modes
        sink_loss_w: float = 0.5,
        class_loss_w: float = 1.0,
        cont_loss_w: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pc_fourier_modes = pc_fourier_modes
        self.sink_loss_w = sink_loss_w
        self.class_loss_w = class_loss_w
        self.cont_loss_w = cont_loss_w


class RaincoatConfig(RaincoatConfigBase):
    """
    Defaults are taking from the hparams used for HAR in the original implementation
    """

    model_type: str = "Raincoat"

    input_len: int  # length of the time series
    n_input_channels: int  # number of input channels
    n_mid_channels: int  # number of mid channels for CNN backbone
    n_out_channels: int  # number of output channels for CNN backbone and for Spectral Convolutional NN
    dropout_p: float  # dropout probability for CNN backbone
    kernel_size: int  # kernel size for Convolutional layers for CNN backbone
    stride: int  # stride for Convolutional layers for CNN backbone
    averagepool_to: int  # length of the output of the CNN backbone

    def __init__(
        self,
        input_len: int = 128,
        n_input_channels: int = 9,
        n_mid_channels: int = 64,
        n_out_channels: int = 192,
        dropout_p: float = 0.5,
        kernel_size: int = 5,
        stride: int = 1,
        averagepool_to: int = 1,
        **args,
    ):
        super().__init__(**args)

        self.input_len = input_len
        self.n_input_channels = n_input_channels
        self.n_mid_channels = n_mid_channels
        self.n_out_channels = n_out_channels
        self.dropout_p = dropout_p
        self.kernel_size = kernel_size
        self.stride = stride
        self.averagepool_to = averagepool_to

    @property
    def n_fourier_modes(self):
        return int(self.input_len * self.pc_fourier_modes / 2 + 1)


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_fourier_modes: int):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fourier_modes = (
            n_fourier_modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, n_fourier_modes, dtype=torch.cfloat)
        )
        self.pi = torch.acos(torch.zeros(1)).item() * 2

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):

        batch_size = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x = torch.cos(x)
        x_ft = torch.fft.rfft(x, norm="ortho")
        out_ft = torch.zeros(
            batch_size, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat
        )
        out_ft[:, :, : self.n_fourier_modes] = self.compl_mul1d(
            x_ft[:, :, : self.n_fourier_modes], self.weights1
        )
        r = out_ft[:, :, : self.n_fourier_modes].abs()  # amplitude
        p = out_ft[:, :, : self.n_fourier_modes].angle()  # phase
        return r, out_ft  # This differs from the paper, where both r and p are returned


class RaincoatBackbone(PreTrainedModel):

    config_class = RaincoatConfig
    config: RaincoatConfig

    conv_block1: nn.Sequential  # First convolutional layer
    dropout: nn.Dropout  # Dropout layer
    conv_block2: nn.Sequential  # Second convolutional layer
    conv_block3: nn.Sequential  # Third convolutional layer
    adaptive_pool: nn.AdaptiveAvgPool1d  # Adaptive pool layer setting the output length

    def __init__(self, config: RaincoatConfig):
        super(RaincoatBackbone, self).__init__(config)

        def Conv_block(
            in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int
        ) -> nn.Sequential:

            conv_block = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    bias=False,
                    padding=padding,
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            )

            return conv_block

        self.conv_block1 = Conv_block(
            self.config.n_input_channels,
            self.config.n_mid_channels,
            self.config.kernel_size,
            self.config.stride,
            padding=(self.config.kernel_size // 2),
        )

        self.dropout = nn.Dropout(self.config.dropout_p)

        self.conv_block2 = Conv_block(
            self.config.n_mid_channels, self.config.n_mid_channels, 8, 1, padding=4
        )

        self.conv_block3 = Conv_block(
            self.config.n_mid_channels, self.config.n_out_channels, 8, 1, padding=4
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.config.averagepool_to)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.dropout(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        x_flat = x.reshape(x.shape[0], -1)
        return (x_flat,)


class Tf_encoder(nn.Module):

    spectralConv: nn.Module  # Frequency domain analyzer
    conv1: nn.Conv1d  # Convolutional layer
    adaptive_pool: nn.AdaptiveAvgPool1d  # Adaptive pool layer setting the output length
    nn2: nn.LayerNorm  # Normalization layer

    def __init__(self, config: RaincoatConfig):
        super(Tf_encoder, self).__init__()
        self.config = config
        self.spectralConv = SpectralConv1d(
            self.config.n_input_channels, self.config.n_input_channels, self.config.n_fourier_modes
        )
        self.conv1 = nn.Conv1d(
            self.config.n_input_channels,
            1,
            kernel_size=3,
            stride=self.config.stride,
            bias=False,
            padding=1,
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.config.n_fourier_modes)
        self.nn2 = nn.LayerNorm(self.config.n_fourier_modes)

    def forward(self, x, x_backboned):

        ef, out_ft = self.spectralConv(x)
        ef = self.nn2(self.adaptive_pool(self.conv1(ef).view((-1, ef.shape[2]))))
        et = x_backboned
        f = torch.concat([ef, et], -1)
        return F.normalize(f), out_ft


class Classifier(nn.Module):
    def __init__(self, input_channels: int, num_classes: int):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(input_channels, num_classes, bias=False)
        self.temperature = 0.05

    def forward(self, x):
        return self.linear(x) / self.temperature


# Adapted from https://github.com/gpeyre/SinkhornAutoDiff


class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps, max_iter, reduction="none"):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):

        device = x.device

        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]

        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = (
            torch.empty(batch_size, x_points, dtype=torch.float, requires_grad=False)
            .fill_(1.0 / x_points)
            .squeeze()
            .to(device)
        )
        nu = (
            torch.empty(batch_size, y_points, dtype=torch.float, requires_grad=False)
            .fill_(1.0 / y_points)
            .squeeze()
            .to(device)
        )

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = (
                self.eps
                * (
                    torch.log(nu + 1e-8)
                    - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)
                )
                + v
            )
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == "mean":
            cost = cost.mean()
        elif self.reduction == "sum":
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"

        return (-C + u.view((-1, 1)) + v.view((1, -1))) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1
