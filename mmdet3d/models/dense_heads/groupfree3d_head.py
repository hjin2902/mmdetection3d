# import numpy as np
import torch
# from mmcv.runner import force_fp32
from torch import nn as nn
from torch.nn import functional as F

# from mmdet3d.core.post_processing import aligned_3d_nms
from mmdet3d.models.builder import build_loss
# from mmdet3d.models.losses import chamfer_distance
from mmdet3d.ops import Points_Sampler, gather_points
from mmdet.core import build_bbox_coder
from mmdet.models import HEADS
from mmdet.models.utils import TransformerDecoderLayer
from .base_conv_bbox_head import BaseConvBboxHead


class PointsObjClsModule(nn.Module):

    def __init__(self, seed_feature_dim):
        """object candidate point prediction from seed point features.

        Args:
            seed_feature_dim: int
                number of channels of seed point features
        """
        super().__init__()
        self.in_dim = seed_feature_dim
        self.conv1 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.bn1 = torch.nn.BatchNorm1d(self.in_dim)
        self.conv2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.bn2 = torch.nn.BatchNorm1d(self.in_dim)
        self.conv3 = torch.nn.Conv1d(self.in_dim, 1, 1)

    def forward(self, seed_features):
        """Forward pass.

        Arguments:
            seed_features: (batch_size, feature_dim, num_seed) Pytorch tensor
        Returns:
            logits: (batch_size, 1, num_seed)
        """
        net = F.relu(self.bn1(self.conv1(seed_features)))
        net = F.relu(self.bn2(self.conv2(net)))
        logits = self.conv3(net)  # (batch_size, 1, num_seed)

        return logits


class GeneralSamplingModule(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, xyz, features, sample_inds):
        """
        Args:
            xyz: (B, N, 3)
            features: (B, C, N)
            sample_inds: (B, M)
        """
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = gather_points(xyz_flipped, sample_inds).transpose(
            1, 2).contiguous()  # (B, M, 3)
        new_features = gather_points(features,
                                     sample_inds).contiguous()  # (B, C, M)

        return new_xyz, new_features, sample_inds


class PositionEmbeddingLearned(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats), nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


@HEADS.register_module()
class GroupFree3DHead(nn.Module):
    r"""Bbox head of `Group-Free 3D https://arxiv.org/abs/2104.00678>`_.

    Args:
        num_classes (int): The number of class.
        bbox_coder (:obj:`BaseBBoxCoder`): Bbox coder for encoding and
            decoding boxes.
        train_cfg (dict): Config for training.
        test_cfg (dict): Config for testing.
        vote_module_cfg (dict): Config of VoteModule for point-wise votes.
        vote_aggregation_cfg (dict): Config of vote aggregation layer.
        pred_layer_cfg (dict): Config of classfication and regression
            prediction layers.
        conv_cfg (dict): Config of convolution in prediction layer.
        norm_cfg (dict): Config of BN in prediction layer.
        objectness_loss (dict): Config of objectness loss.
        center_loss (dict): Config of center loss.
        dir_class_loss (dict): Config of direction classification loss.
        dir_res_loss (dict): Config of direction residual regression loss.
        size_class_loss (dict): Config of size classification loss.
        size_res_loss (dict): Config of size residual regression loss.
        semantic_loss (dict): Config of point-wise semantic segmentation loss.
    """

    def __init__(
            self,
            num_classes,
            bbox_coder,
            transformer_cfg,
            train_cfg=None,
            test_cfg=None,
            #  vote_module_cfg=None,
            #  vote_aggregation_cfg=None,
            num_proposal=128,
            pred_layer_cfg=None,
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d'),
            objectness_loss=None,
            center_loss=None,
            dir_class_loss=None,
            dir_res_loss=None,
            size_class_loss=None,
            size_res_loss=None,
            semantic_loss=None,
            iou_loss=None):
        super(GroupFree3DHead, self).__init__()
        self.num_classes = num_classes
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # self.gt_per_seed = vote_module_cfg['gt_per_seed']
        # self.num_proposal = vote_aggregation_cfg['num_point']
        self.num_proposal = num_proposal

        # transformer cfg
        self.num_decoder_layers = transformer_cfg['num_decoder_layers']
        self.self_pos_embed = transformer_cfg['self_pos_embed']
        self.cross_pos_embed = transformer_cfg['cross_pos_embed']
        self.d_model = transformer_cfg['d_model']

        self.nhead = transformer_cfg['nhead']
        self.dim_feedforward = transformer_cfg['dim_feedforward']
        self.dropout = transformer_cfg['dropout']

        self.objectness_loss = build_loss(objectness_loss)
        self.center_loss = build_loss(center_loss)
        self.dir_res_loss = build_loss(dir_res_loss)
        self.dir_class_loss = build_loss(dir_class_loss)
        self.size_res_loss = build_loss(size_res_loss)
        if size_class_loss is not None:
            self.size_class_loss = build_loss(size_class_loss)
        if semantic_loss is not None:
            self.semantic_loss = build_loss(semantic_loss)
        if iou_loss is not None:
            self.iou_loss = build_loss(iou_loss)
        else:
            self.iou_loss = None

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.num_sizes = self.bbox_coder.num_sizes
        self.num_dir_bins = self.bbox_coder.num_dir_bins

        # Initial object candidate sampling
        self.gsample_module = GeneralSamplingModule()
        if self.sampling == 'fps':
            self.fps_module = Points_Sampler(num_proposal)
        elif self.sampling == 'kps':
            self.points_obj_cls = PointsObjClsModule(self.d_model)
        else:
            raise NotImplementedError

        # self.vote_module = VoteModule(**vote_module_cfg)
        # self.vote_aggregation = build_sa_module(vote_aggregation_cfg)
        self.fp16_enabled = False

        # Bbox classification and regression
        self.conv_pred = BaseConvBboxHead(
            **pred_layer_cfg,
            num_cls_out_channels=self._get_cls_out_channels(),
            num_reg_out_channels=self._get_reg_out_channels())

        # Transformer Decoder Projection
        self.decoder_key_proj = nn.Conv1d(
            self.d_model, self.d_model, kernel_size=1)
        self.decoder_query_proj = nn.Conv1d(
            self.d_model, self.d_model, kernel_size=1)

        # Position Embedding for Self-Attention
        if self.self_pos_embed == 'none':
            self.decoder_self_posembeds = [
                None for i in range(self.num_decoder_layers)
            ]
        elif self.self_pos_embed == 'xyz_learned':
            self.decoder_self_posembeds = nn.ModuleList()
            for i in range(self.num_decoder_layers):
                self.decoder_self_posembeds.append(
                    PositionEmbeddingLearned(3, self.d_model))
        elif self.self_pos_embed == 'loc_learned':
            self.decoder_self_posembeds = nn.ModuleList()
            for i in range(self.num_decoder_layers):
                self.decoder_self_posembeds.append(
                    PositionEmbeddingLearned(6, self.d_model))
        else:
            raise NotImplementedError(f'self_position_embedding not supported \
                    {self.self_position_embedding}')

        # Position Embedding for Cross-Attention
        if self.cross_pos_embed == 'none':
            self.decoder_cross_posembeds = [
                None for i in range(self.num_decoder_layers)
            ]
        elif self.cross_pos_embed == 'xyz_learned':
            self.decoder_cross_posembeds = nn.ModuleList()
            for i in range(self.num_decoder_layers):
                self.decoder_cross_posembeds.append(
                    PositionEmbeddingLearned(3, self.d_model))
        else:
            raise NotImplementedError(
                f'cross_position_embedding not supported \
                    {self.cross_pos_embed}')

        # Transformer decoder layers
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(self.d_model, self.nhead,
                                        self.dim_feedforward, self.dropout))

        # Prediction Head
        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.prediction_heads.append(
                BaseConvBboxHead(
                    **pred_layer_cfg,
                    num_cls_out_channels=self._get_cls_out_channels(),
                    num_reg_out_channels=self._get_reg_out_channels()))

    def init_weights(self):
        """Initialize weights of GroupFree3DHead."""
        pass

    def _get_cls_out_channels(self):
        """Return the channel number of classification outputs."""
        # Class numbers (k) + objectness (1)
        return self.num_classes + 1

    def _get_reg_out_channels(self):
        """Return the channel number of regression outputs."""
        # Objectness scores (1), center residual (3),
        # heading class+residual (num_dir_bins*2),
        # size class+residual(num_sizes*4)
        return 3 + self.num_dir_bins * 2 + self.num_sizes * 4

    def _extract_input(self, feat_dict):
        """Extract inputs from features dictionary.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            torch.Tensor: Coordinates of input points.
            torch.Tensor: Features of input points.
            torch.Tensor: Indices of input points.
        """

        # for imvotenet
        if 'seed_points' in feat_dict and \
           'seed_features' in feat_dict and \
           'seed_indices' in feat_dict:
            seed_points = feat_dict['seed_points']
            seed_features = feat_dict['seed_features']
            seed_indices = feat_dict['seed_indices']
        # for votenet
        else:
            seed_points = feat_dict['fp_xyz'][-1]
            seed_features = feat_dict['fp_features'][-1]
            seed_indices = feat_dict['fp_indices'][-1]

        return seed_points, seed_features, seed_indices

    def forward(self, feat_dict, sample_mod):
        """Forward pass.

        Note:
            The forward of GroupFree3DHead is devided into 2 steps:

                1. Initial object candidates sampling.
                2. Iterative object box prediction by transformer decoder.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            results (dict): Predictions of GroupFree3D head.
        """
        assert sample_mod in ['fps', 'kps']

        seed_points, seed_features, seed_indices = self._extract_input(
            feat_dict)

        results = dict(
            seed_points=seed_points,
            seed_features=seed_features,
            seed_indices=seed_indices)

        # for key and key_pos in Transformer Decoder
        points_xyz = feat_dict['fp_xyz'][-1]
        points_features = feat_dict['fp_features'][-1]

        # 1. Initial object candidates sampling.
        if sample_mod == 'fps':
            sample_inds = self.fps_module(seed_points, seed_features)
        elif sample_mod == 'kps':
            points_obj_cls_logits = self.points_obj_cls(
                seed_features)  # (batch_size, 1, num_seed)
            points_obj_cls_scores = torch.sigmoid(
                points_obj_cls_logits).squeeze(1)
            sample_inds = torch.topk(points_obj_cls_scores,
                                     self.num_proposal)[1].int()
            results['seeds_obj_cls_logits'] = points_obj_cls_logits
        else:
            raise NotImplementedError(
                f'Sample mode {sample_mod} is not supported!')

        xyz, features, sample_inds = self.gsample_module(
            seed_points, seed_features, sample_inds)

        cluster_feature = features
        cluster_xyz = xyz
        results['query_points_xyz'] = xyz  # (B, M, 3)
        results['query_points_feature'] = features  # (B, C, M)
        results['query_points_sample_inds'] = sample_inds  # (B, M)

        cls_predictions, reg_predictions = self.conv_pred(cluster_feature)
        decode_res = self.bbox_coder.split_pred(cls_predictions,
                                                reg_predictions, cluster_xyz)

        results.update(decode_res)
        bbox3d = self.bbox_coder.decode(results)

        # 2. Iterative object box prediction by transformer decoder.
        base_xyz = bbox3d[:, :, :3].detach().clone()
        base_size = bbox3d[:, :, 3:6].detach().clone()

        # Transformer Decoder and Prediction
        query = self.decoder_query_proj(cluster_feature)
        key = self.decoder_key_proj(points_features)

        # Position Embedding for Cross-Attention
        if self.cross_pos_embed == 'none':
            key_pos = None
        elif self.cross_pos_embed in ['xyz_learned']:
            key_pos = points_xyz
        else:
            raise NotImplementedError(
                f'cross_position_embedding not supported \
                    {self.cross_pos_embed}')

        for i in range(self.num_decoder_layers):

            # Position Embedding for Self-Attention
            if self.self_pos_embed == 'none':
                query_pos = None
            elif self.self_pos_embed == 'xyz_learned':
                query_pos = base_xyz
            elif self.self_pos_embed == 'loc_learned':
                query_pos = torch.cat([base_xyz, base_size], -1)
            else:
                raise NotImplementedError(
                    f'self_position_embedding not supported \
                        {self.self_pos_embed}')

            query_pos_embed = self.decoder_self_posembeds[i](
                query_pos).permute(2, 0, 1)
            key_pos_embed = self.decoder_cross_posembeds[i](key_pos).permute(
                2, 0, 1)

            # Transformer Decoder Layer
            query = self.decoder[i](query, key, query_pos_embed, key_pos_embed)

            # Prediction
            cls_predictions, reg_predictions = self.prediction_heads[i](query)

            decode_res = self.bbox_coder.split_pred(cls_predictions,
                                                    reg_predictions,
                                                    cluster_xyz)

            results.update(decode_res)

            bbox3d = self.bbox_coder.decode(results)

            base_xyz = bbox3d[:, :, :3].detach().clone()
            base_size = bbox3d[:, :, 3:6].detach().clone()

        return results
