import torch
import copy
from torch import nn
import numpy as np
import torch.nn.functional as F
from _classifiers.hf.sasa.sparsemax import Sparsemax
from transformers import PreTrainedModel
from typing import Type, TypeVar, Dict
from _classifiers.hf.base import HuggingFaceClassifier
from _datasets.base import TLDataset
from _classifiers.hf.common import CommonTrainer
from _utils.enumerations import *
from _backbone.nn.sasa import SASAConfigBase

THFSASABase = TypeVar("THFSASABase", bound="HFSASABase")


class SASABasemodel(PreTrainedModel):
    """
    Abstract base model for sasa based models.
    Not intended as a standalone class.
    """

    config: SASAConfigBase

    def __init__(self, config: SASAConfigBase, backbone: PreTrainedModel):
        super().__init__(config)
        self.backbone = backbone

        self.sparse_max = Sparsemax(dim=-1)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.max_len = self.config.window_size
        self.segments_num = self.config.segments_num
        self.feature_dim = self.config.n_input_channels
        self.h_dim = self.config.h_dim
        self.dense_dim = self.config.dense_dim
        self.drop_prob = self.config.drop_prob
        self.class_num = self.config.num_labels
        self.coeff = self.config.coeff
        self.base_bone_list = nn.ModuleList(
            [copy.deepcopy(backbone) for _ in range(0, self.feature_dim)]
        )
        self.self_attn_Q = nn.Sequential(
            nn.Linear(in_features=self.h_dim, out_features=self.h_dim), nn.ELU()
        )

        self.self_attn_K = nn.Sequential(
            nn.Linear(in_features=self.h_dim, out_features=self.h_dim), nn.LeakyReLU()
        )
        self.self_attn_V = nn.Sequential(
            nn.Linear(in_features=self.h_dim, out_features=self.h_dim), nn.LeakyReLU()
        )

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.feature_dim * 2 * self.h_dim),
            nn.Linear(self.feature_dim * 2 * self.h_dim, self.dense_dim),
            nn.BatchNorm1d(self.dense_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.dense_dim, self.class_num),
        )
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, mts, labels, domain_y=None):
        if domain_y is not None:
            return self.forward_train(mts, labels, domain_y)
        else:
            return self.forward_eval(mts, labels)

    def forward_eval(self, mts, labels):
        x = mts
        feature, src_intra_aw_list, src_inter_aw_list = self.calculate_feature_alpha_beta(x)
        logits = self.classifier(feature)
        y = labels
        cls_loss = self.cross_entropy(logits, y)
        return cls_loss, logits

    def forward_train(self, mts, labels, domain_y):
        # Source and target indices
        src_indxs = domain_y == DomainEnumInt.source.value
        trg_indxs = domain_y == DomainEnumInt.target.value

        src_x = mts[src_indxs]
        tgt_x = mts[trg_indxs]

        if tgt_x.shape[0] == 0:
            tgt_feature = None
        else:
            tgt_feature, tgt_intra_aw_list, tgt_inter_aw_list = self.calculate_feature_alpha_beta(
                tgt_x
            )

        src_feature, src_intra_aw_list, src_inter_aw_list = self.calculate_feature_alpha_beta(src_x)

        domain_loss_alpha = []
        domain_loss_beta = []

        input_clf_shape = list(src_feature.shape)
        input_clf_shape[0] = mts.shape[0]
        input_clf = torch.zeros(*input_clf_shape, device=src_feature.device)
        input_clf[src_indxs] = src_feature
        if tgt_feature is not None:
            input_clf[trg_indxs] = tgt_feature

        logits = self.classifier(input_clf)

        if (src_x.shape[0] == tgt_x.shape[0]) and (src_x.shape[0] > 1):
            # only do uda if same support for source and target
            for i in range(self.feature_dim):
                domain_loss_intra = self.mmd_loss(
                    src_struct=src_intra_aw_list[i],
                    tgt_struct=tgt_intra_aw_list[i],
                    weight=self.coeff,
                )
                domain_loss_inter = self.mmd_loss(
                    src_struct=src_inter_aw_list[i],
                    tgt_struct=tgt_inter_aw_list[i],
                    weight=self.coeff,
                )
                domain_loss_alpha.append(domain_loss_intra)
                domain_loss_beta.append(domain_loss_inter)

            total_domain_loss_alpha = torch.tensor(domain_loss_alpha).mean()
            total_domain_loss_beta = torch.tensor(domain_loss_beta).mean()

        else:
            total_domain_loss_alpha = 0
            total_domain_loss_beta = 0

        src_cls_loss = self.cross_entropy(logits, labels)
        total_loss = src_cls_loss + total_domain_loss_beta + total_domain_loss_alpha
        return total_loss, logits

    def self_attention(self, Q, K, scale=True, sparse=True, k=3):

        segment_num = Q.shape[1]

        attention_weight = torch.bmm(Q, K.permute(0, 2, 1))

        attention_weight = torch.mean(attention_weight, dim=2, keepdim=True)

        if scale:
            d_k = torch.tensor(K.shape[-1]).float()
            attention_weight = attention_weight / torch.sqrt(d_k)
        if sparse:
            attention_weight_sparse = self.sparse_max(
                torch.reshape(attention_weight, [-1, segment_num])
            )
            attention_weight = torch.reshape(
                attention_weight_sparse, [-1, attention_weight.shape[1], attention_weight.shape[2]]
            )
        else:
            attention_weight = self.softmax(attention_weight)

        return attention_weight

    def attention_fn(self, Q, K, scaled=False, sparse=True, k=1):
        segment_num = Q.shape[1]

        attention_weight = torch.matmul(
            F.normalize(Q, p=2, dim=-1), F.normalize(K, p=2, dim=-1).permute(0, 1, 3, 2)
        )

        if scaled:
            d_k = torch.tensor(K.shape[-1]).float()
            attention_weight = attention_weight / torch.sqrt(d_k)
            attention_weight = (
                k * torch.log(torch.tensor(segment_num, dtype=torch.float32)) * attention_weight
            )

        if sparse:
            attention_weight_sparse = self.sparse_max(
                torch.reshape(attention_weight, [-1, self.segments_num])
            )

            attention_weight = torch.reshape(attention_weight_sparse, attention_weight.shape)
        else:
            attention_weight = self.softmax(attention_weight)

        return attention_weight

    def mmd_loss(self, src_struct, tgt_struct, weight):
        delta = torch.mean(src_struct - tgt_struct, dim=-2)
        loss_value = torch.norm(delta, 2) * weight
        return loss_value

    def calculate_feature_alpha_beta(self, x):
        uni_candidate_representation_list = {}

        uni_adaptation_representation = {}

        intra_attn_weight_list = {}
        inter_attn_weight_list = {}

        Hi_list = []

        for i in range(0, self.feature_dim):
            xi = torch.reshape(x[:, :, i, :], shape=[-1, self.max_len, 1])
            _, (candidate_representation_xi, _) = self.base_bone_list[i](xi)[0]

            candidate_representation_xi = torch.reshape(
                candidate_representation_xi, shape=[-1, self.segments_num, self.h_dim]
            )

            uni_candidate_representation_list[i] = candidate_representation_xi

            Q_xi = self.self_attn_Q(candidate_representation_xi)
            K_xi = self.self_attn_K(candidate_representation_xi)
            V_xi = self.self_attn_V(candidate_representation_xi)

            intra_attention_weight_xi = self.self_attention(Q=Q_xi, K=K_xi, sparse=True)

            Z_i = torch.bmm(
                intra_attention_weight_xi.view(intra_attention_weight_xi.shape[0], 1, -1), V_xi
            )

            intra_attn_weight_list[i] = torch.squeeze(intra_attention_weight_xi)
            Z_i = F.normalize(Z_i, dim=-1)

            uni_adaptation_representation[i] = Z_i

        for i in range(0, self.feature_dim):
            Z_i = uni_adaptation_representation[i]
            other_candidate_representation_src = torch.stack(
                [uni_candidate_representation_list[j] for j in range(self.feature_dim)], dim=0
            )

            inter_attention_weight = self.attention_fn(
                Q=Z_i, K=other_candidate_representation_src, sparse=True
            )

            U_i_src = torch.mean(
                torch.matmul(inter_attention_weight, other_candidate_representation_src), dim=0
            )

            inter_attn_weight_list[i] = torch.squeeze(inter_attention_weight)
            Hi = torch.squeeze(torch.cat([Z_i, U_i_src], dim=-1), dim=1)
            Hi = F.normalize(Hi, dim=-1)
            Hi_list.append(Hi)
        final_feature = torch.reshape(
            torch.cat(
                Hi_list,
                dim=-1,
            ),
            shape=[x.shape[0], self.feature_dim * 2 * self.h_dim],
        )
        return final_feature, intra_attn_weight_list, inter_attn_weight_list


class HFSASABase(HuggingFaceClassifier):
    def __init__(
        self,
        config: Dict,
        pretrained_mdl_cls: Type[SASABasemodel],
        pretrained_cfg_cls: Type[SASAConfigBase],
        **kwargs,
    ):
        super().__init__(
            config=config,
            pretrained_mdl_cls=pretrained_mdl_cls,
            pretrained_cfg_cls=pretrained_cfg_cls,
            **kwargs,
        )

        # Optimizer used in Raincoat
        optimizer = torch.optim.AdamW(
            [p for p in self.classifier.parameters() if p.requires_grad],
            lr=self.config.training.get("learning_rate"),
            weight_decay=self.config.training.get("weight_decay"),
        )
        self.addition_training_args["optimizers"] = (optimizer, None)

        self.trainer_class = CommonTrainer

    def get_sasa_format_tl_dataset(self, tl_dataset: TLDataset) -> TLDataset:
        # will transform time series to a set of subseries

        def extract_sub_seq(ex: dict) -> dict:
            window_size = self.classifier.config.window_size
            mts = ex[DatasetColumnsEnum.mts]
            sample = []
            for length in self.classifier.config.segments_length:
                a = mts[:, (-length):]  # [channels, seq_length]
                a = np.pad(
                    a, pad_width=((0, 0), (0, window_size - length)), mode="constant"
                )  # padding to [channels, window_size]
                sample.append(a)

            sample = np.array(sample)

            # keep original to be used with gmm later on
            ex[DatasetColumnsEnum.ori_mts] = mts

            ex[DatasetColumnsEnum.mts] = sample
            return ex

        new_source = tl_dataset.source.map(lambda example: extract_sub_seq(example))
        new_target = tl_dataset.target.map(lambda example: extract_sub_seq(example))

        sasa_tl_dataset = TLDataset(
            dataset_name=tl_dataset.dataset_name,
            preprocessing_pipeline=tl_dataset.preprocessing_pipeline,
            source=new_source,
            target=new_target,
            gmm_root_dir=tl_dataset.gmm_root_dir,
        )

        return sasa_tl_dataset

    def fit(self, tl_dataset: TLDataset) -> THFSASABase:
        new_tl_dataset = self.get_sasa_format_tl_dataset(self.get_new_tl_dataset(tl_dataset))

        return super().fit(new_tl_dataset)

    def predict(self, tl_dataset: TLDataset):
        new_tl_dataset = self.get_sasa_format_tl_dataset(tl_dataset)
        return super().predict(new_tl_dataset)

    def inject_config_based_on_dataset(self, config: Dict, tl_dataset: TLDataset) -> Dict:
        # check if attr exist
        if InjectConfigEnum.backbone not in config:
            config[InjectConfigEnum.backbone] = {}
        # add num channels
        config[InjectConfigEnum.backbone][InjectConfigEnum.n_input_channels] = len(
            tl_dataset.source[DataSplitEnum.train][DatasetColumnsEnum.mts][0]
        )
        return super().inject_config_based_on_dataset(config, tl_dataset)
