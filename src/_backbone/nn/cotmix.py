from transformers import PreTrainedModel, PretrainedConfig
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

"""
Feature Extractor and Classifier for the CoTMix domain adaptation framework as described in https://arxiv.org/abs/2212.01555.
Part of the code is taken directly from https://github.com/emadeldeen24/CoTMix, other parts are re-implemented for compatibility with our framework.
"""


class ConditionalEntropyLoss(nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.2, contrast_mode="all"):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        # Assigning device as the device of the inputs
        device = features.device

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...]," "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = -self.temperature * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class NTXentLoss(nn.Module):
    def __init__(self, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=-1)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        # Verify that both inputs are on the same device
        assert zis.device == zjs.device
        # Assigning device as the device of the inputs
        device = zis.device

        # Get batch size
        batch_size = zis.shape[0]

        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        mask_samples_from_same_repr = mask.to(device)

        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        negatives = similarity_matrix[mask_samples_from_same_repr].view(2 * batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * batch_size).to(device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * batch_size)


class MixConfig(PretrainedConfig):
    """Common config interface for all models implementing mix approach"""

    temperature: float  # temperature parameter for loss functions
    use_cosine_similarity: bool  # whether or not to use the cosine similary distance for NTXentLoss
    temporal_shift: int  # determines the size of the window for temporal mixture
    mix_ratio: float  # percentage of the natural sample in the mixture
    # Weights for the linear combination of loss functions:
    beta1: float  # weight for source classification loss (src_cls_weight in original implementation)
    beta2: float  # weight for traget entropy loss (trg_entropy_weight in original implementation)
    beta3: float  # weight for source supervised contrastive loss (src_supCon_weight in original implementation)
    beta4: float  # weight for target contrastive loss (trg_cont_weight in original implementation)

    def __init__(
        self,
        temperature: float = 0.2,
        use_cosine_similarity: bool = True,
        mix_ratio: float = 0.9,
        temporal_shift: int = 10,
        beta1: float = 0.9,
        beta2: float = 0.05,
        beta3: float = 0.1,
        beta4: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity
        self.mix_ratio = mix_ratio
        self.temporal_shift = temporal_shift
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.beta4 = beta4


class CoTMixConfig(MixConfig):

    model_type: str = "CoTMix"

    n_input_channels: int  # number of input channels
    n_mid_channels: int  # number of channels at the output of the first Conv1d block
    kernel_size: int  # kernel size of the first Conv1d layer
    stride: int  # stride of first Conv1d layer
    dropout: float  # p parameter for dropout layer
    n_out_channels: int  # number of output channels
    avgpool_to: int  # output length for the feature extractor

    def __init__(
        self,
        n_input_channels: int = 1,
        n_mid_channels: int = 64,
        kernel_size: int = 5,
        stride: int = 1,
        dropout: float = 0.5,
        n_out_channels: int = 2,
        avgpool_to: int = 32,
        **args,
    ):
        super().__init__(**args)

        self.n_input_channels = n_input_channels
        self.n_mid_channels = n_mid_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout = dropout
        self.n_out_channels = n_out_channels
        self.avgpool_to = avgpool_to


class CoTMixBackbone(PreTrainedModel):

    config_class = CoTMixConfig
    config: CoTMixConfig

    conv_block1: nn.Sequential
    dropout: nn.Dropout
    conv_block2: nn.Sequential
    conv_block3: nn.Sequential
    adaptive_pool: nn.AdaptiveAvgPool1d

    def __init__(self, config: CoTMixConfig):
        super(CoTMixBackbone, self).__init__(config)

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
            self.config.kernel_size // 2,
        )

        self.dropout = nn.Dropout(config.dropout)

        self.conv_block2 = Conv_block(
            self.config.n_mid_channels, self.config.n_mid_channels * 2, 8, 1, 4
        )

        self.conv_block3 = Conv_block(
            self.config.n_mid_channels * 2, self.config.n_out_channels, 8, 1, 4
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.config.avgpool_to)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.dropout(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)

        x_flat = x.reshape(x.shape[0], -1)
        return (x_flat,)


class Classifier(nn.Module):

    linear: nn.Linear

    def __init__(self, avgpool_to: int, n_out_channels: int, num_labels: int):
        super().__init__()
        self.linear = nn.Linear(avgpool_to * n_out_channels, num_labels)

    def forward(self, x):
        return self.linear(x)
