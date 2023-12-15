from transformers import PreTrainedModel, PretrainedConfig
import torch
from torch import nn
from typing import List
from _utils.enumerations import TorchDeviceEnum

# Constant for numerical stability
EPS = torch.Tensor([torch.finfo(torch.float).eps])  # numerical logs
# Pi constant
PI = torch.Tensor([torch.pi])


class VRADAConfig(PretrainedConfig):

    model_type: str = "VRADA"

    n_input_channels: int  # Number of features for timestep. 'x' must be of shape [N,sequence_length,n_input_features], where N is the number of samples.
    VRNN_h_dim: int  # Size of VRNN hidden dimension
    VRNN_z_dim: int  # Size of latent dimension
    VRNN_n_layers: int  # Number of layers for the VRNN
    Classifier_h_dim: int  # Size of source task classifier hidden layer
    Discrimiator_h_dim: int  # Size of discriminator hidden layer
    R_loss_w: float  # Weight for the reconstruction loss
    num_source_domains: int  # Number of source domains

    def __init__(
        self,
        n_input_channels: int = 3,
        VRNN_h_dim: int = 100,
        VRNN_z_dim: int = 100,
        VRNN_n_layers: int = 1,
        Classifier_h_dim: int = 64,
        Discrimiator_h_dim: int = 64,
        R_loss_w: float = 0.0001,
        num_source_domains: int = 1,
        **args,
    ):
        super().__init__(**args)
        self.n_input_channels = n_input_channels
        self.VRNN_h_dim = VRNN_h_dim
        self.VRNN_z_dim = VRNN_z_dim
        self.VRNN_n_layers = VRNN_n_layers
        self.Classifier_h_dim = Classifier_h_dim
        self.Discrimiator_h_dim = Discrimiator_h_dim
        self.num_source_domains = num_source_domains
        self.R_loss_w = R_loss_w


class VRNN(PreTrainedModel):
    # Inspired from https://github.com/emited/VariationalRecurrentNeuralNetwork/blob/master/model.py

    """implementation of the Variational Recurrent Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
    using unimodal isotropic gaussian distributions for inference, prior, and generating models.
    """

    config_class = VRADAConfig
    config: VRADAConfig

    EPS: torch.Tensor
    PI: torch.Tensor
    phi_x: nn.Sequential
    phi_z: nn.Sequential
    encoder: nn.Sequential
    enc_mean: nn.Linear
    enc_std: nn.Sequential
    decoder: nn.Sequential
    dec_mean: nn.Linear
    dec_std: nn.Sequential
    prior: nn.Sequential
    prior_mean: nn.Linear
    prior_std: nn.Sequential
    recurrence: nn.GRU

    def __init__(self, config: VRADAConfig):
        super().__init__(config)

        h_dim = self.config.VRNN_h_dim

        # Feature extractors for the input and latent representation
        self.phi_x = nn.Sequential(
            nn.Linear(self.config.n_input_channels, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )

        self.phi_z = nn.Sequential(
            nn.Linear(self.config.VRNN_z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )

        # The encoder maps the extrated features from the input and the last hidden state into the latent representation
        self.encoder = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )

        # Mean and std to be used with the re-parametrization trick
        self.enc_mean = nn.Linear(h_dim, self.config.VRNN_z_dim)

        self.enc_std = nn.Sequential(nn.Linear(h_dim, self.config.VRNN_z_dim), nn.Softplus())

        # Prior used to compute the VRNN objective function
        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim), nn.ReLU()
        )

        # Mean and std used to compute the KL divergence between prior and conditional distributions
        self.prior_mean = nn.Linear(h_dim, self.config.VRNN_z_dim)

        self.prior_std = nn.Sequential(nn.Linear(h_dim, self.config.VRNN_z_dim), nn.Softplus())

        # Decoder used to reconstruct the input
        self.decoder = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )

        self.dec_mean = nn.Sequential(nn.Linear(h_dim, self.config.n_input_channels), nn.Sigmoid())

        self.dec_std = nn.Sequential(nn.Linear(h_dim, self.config.n_input_channels), nn.Softplus())

        # Recurrence used to generate the next hidden state. Its must be passed with sequence length 1.
        self.recurrence = nn.GRU(h_dim + h_dim, h_dim, self.config.VRNN_n_layers)

    def _reparameterize_sample(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Reparametrization trick. Sample from a normal distribution and then multiply by 'std' and add 'mean'."""
        eps = torch.empty(size=std.shape, dtype=torch.float).normal_()
        eps = eps.to(self.device)
        return eps.mul(std).add_(mean)

    def _kld_gauss(
        self,
        mean_1: torch.Tensor,
        std_1: torch.Tensor,
        mean_2: torch.Tensor,
        std_2: torch.Tensor,
    ) -> torch.Tensor:
        """Kullback-Leibler Divergence between two Gaussians."""

        batch_size = mean_1.shape[0]

        kld_element = (
            2 * torch.log(std_2 + self.EPS)
            - 2 * torch.log(std_1 + self.EPS)
            + (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / std_2.pow(2)
            - 1
        )
        return 0.5 * torch.sum(kld_element) / batch_size

    def _nll_gauss(self, mean: torch.Tensor, std: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Minus log of Gaussian distribution."""

        batch_size = x.shape[0]

        return (
            torch.sum(
                torch.log(std + self.EPS)
                + torch.log(2 * self.PI) / 2
                + (x - mean).pow(2) / (2 * std.pow(2))
            )
            / batch_size
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Returns the last latent vector along with the loss necessary to train the encoder and decoder."""

        self.EPS = EPS.to(self.device)
        self.PI = PI.to(self.device)

        kld_loss = 0  # KL-divergence between prior and conditional distribution
        nll_loss = 0  # Reconstruction loss for the decoder

        h = torch.zeros(
            (self.config.VRNN_n_layers, x.shape[0], self.config.VRNN_h_dim),
            device=self.device,
        )

        for t in range(x.shape[2]):  # Loop over the time dimension

            x_t = x[:, :, t]

            phi_x_t = self.phi_x(x_t)

            # encoder
            enc_t = self.encoder(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterize_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.decoder(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            # recurrence
            _, h = self.recurrence(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            # computing losses
            kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x_t)

        kld_loss = kld_loss / x.shape[2]
        nll_loss = nll_loss / x.shape[2]

        return [kld_loss, nll_loss, enc_mean_t]


class Classifier_backbone(nn.Module):
    """Classifier for the source task"""

    h_dim: int
    z_dim: int
    FCnet: nn.Sequential

    def __init__(self, h_dim: int, z_dim: int):
        super().__init__()

        self.h_dim = h_dim  # Hidden layer dimension
        self.z_dim = z_dim  # Latent space dimension

        self.FCnet = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.FCnet(z)


class Discriminator(nn.Module):
    """Critic with the task of distinguishing between source and target domain samples."""

    h_dim: int
    z_dim: int
    FCnet: nn.Sequential

    def __init__(self, h_dim: int, z_dim: int, num_source_domains: int):
        super().__init__()

        self.h_dim = h_dim  # Hidden layer dimension
        self.z_dim = z_dim  # Latent space dimension

        self.FCnet = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, num_source_domains + 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.FCnet(z)
