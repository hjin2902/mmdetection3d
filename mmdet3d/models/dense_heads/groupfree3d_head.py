import copy
import numpy as np
import torch
from mmcv import ConfigDict
from mmcv.cnn.bricks.transformer import build_transformer_layer
from mmcv.runner import force_fp32
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.core.post_processing import aligned_3d_nms
from mmdet3d.models.builder import build_loss
# from mmdet3d.models.losses import chamfer_distance
from mmdet3d.ops import Points_Sampler, gather_points
from mmdet.core import build_bbox_coder, multi_apply
from mmdet.models import HEADS
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
        # torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out')
        # print('self.conv1.weight: ', self.conv1.weight)
        # print(self.conv1.weight.shape)

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
    r"""Bbox head of `Group-Free 3D <https://arxiv.org/abs/2104.00678>`_.

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

    def __init__(self,
                 num_classes,
                 in_channels,
                 bbox_coder,
                 num_decoder_layers,
                 transformerlayers,
                 train_cfg=None,
                 test_cfg=None,
                 num_proposal=128,
                 pred_layer_cfg=None,
                 size_cls_agnostic=True,
                 gt_per_seed=3,
                 sampling_objectness_loss=None,
                 objectness_loss=None,
                 center_loss=None,
                 dir_class_loss=None,
                 dir_res_loss=None,
                 size_class_loss=None,
                 size_res_loss=None,
                 size_reg_loss=None,
                 semantic_loss=None,
                 iou_loss=None):
        super(GroupFree3DHead, self).__init__()
        self.num_classes = num_classes
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_proposal = num_proposal
        self.in_channels = in_channels
        self.num_decoder_layers = num_decoder_layers
        self.size_cls_agnostic = size_cls_agnostic
        self.gt_per_seed = gt_per_seed

        # Transformer decoder layers
        if isinstance(transformerlayers, ConfigDict):
            transformerlayers = [
                copy.deepcopy(transformerlayers)
                for _ in range(num_decoder_layers)
            ]
        else:
            assert isinstance(transformerlayers, list) and \
                   len(transformerlayers) == num_decoder_layers
        self.decoder_layers = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder_layers.append(
                build_transformer_layer(transformerlayers[i]))
        self.embed_dims = self.decoder_layers[0].embed_dims

        # bbox_coder
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.num_sizes = self.bbox_coder.num_sizes
        self.num_dir_bins = self.bbox_coder.num_dir_bins

        # Initial object candidate sampling
        self.gsample_module = GeneralSamplingModule()
        self.fps_module = Points_Sampler([self.num_proposal])
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        self.points_obj_cls = PointsObjClsModule(self.in_channels)

        self.fp16_enabled = False

        # initial candidate prediction
        self.conv_pred = BaseConvBboxHead(
            **pred_layer_cfg,
            num_cls_out_channels=self._get_cls_out_channels(),
            num_reg_out_channels=self._get_reg_out_channels())

        # query proj and key proj
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        self.decoder_query_proj = nn.Conv1d(
            self.embed_dims, self.embed_dims, kernel_size=1)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        self.decoder_key_proj = nn.Conv1d(
            self.embed_dims, self.embed_dims, kernel_size=1)

        # query position embed
        self.decoder_self_posembeds = nn.ModuleList()
        for _ in range(self.num_decoder_layers):
            self.decoder_self_posembeds.append(
                PositionEmbeddingLearned(6, self.embed_dims))
        # key position embed
        self.decoder_cross_posembeds = nn.ModuleList()
        for _ in range(self.num_decoder_layers):
            self.decoder_cross_posembeds.append(
                PositionEmbeddingLearned(3, self.embed_dims))

        # Prediction Head
        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.prediction_heads.append(
                BaseConvBboxHead(
                    **pred_layer_cfg,
                    num_cls_out_channels=self._get_cls_out_channels(),
                    num_reg_out_channels=self._get_reg_out_channels()))

        self.sampling_objectness_loss = build_loss(sampling_objectness_loss)
        self.objectness_loss = build_loss(objectness_loss)
        self.center_loss = build_loss(center_loss)
        self.dir_res_loss = build_loss(dir_res_loss)
        self.dir_class_loss = build_loss(dir_class_loss)
        self.semantic_loss = build_loss(semantic_loss)
        if self.size_cls_agnostic:
            self.size_reg_loss = build_loss(size_reg_loss)
        else:
            self.size_res_loss = build_loss(size_res_loss)
            self.size_class_loss = build_loss(size_class_loss)

    def init_weights(self):
        """Initialize weights of transformer decoder in GroupFree3DHead."""
        # TODO
        # initialize transformer
        for m in self.decoder_layers.parameters():
            if m.dim() > 1:
                torch.manual_seed(0)
                torch.cuda.manual_seed_all(0)
                torch.backends.cudnn.deterministic = True
                nn.init.xavier_uniform_(m)

        for m in self.decoder_self_posembeds.parameters():
            if m.dim() > 1:
                torch.manual_seed(0)
                torch.cuda.manual_seed_all(0)
                torch.backends.cudnn.deterministic = True
                nn.init.xavier_uniform_(m)
            else:
                nn.init.constant_(m, 0)

        for m in self.decoder_cross_posembeds.parameters():
            if m.dim() > 1:
                torch.manual_seed(0)
                torch.cuda.manual_seed_all(0)
                torch.backends.cudnn.deterministic = True
                nn.init.xavier_uniform_(m)
            else:
                nn.init.constant_(m, 0)

        # for m in self.conv_pred.parameters():
        #     if m.dim() > 1:
        #         torch.manual_seed(0)
        #         torch.cuda.manual_seed_all(0)
        #         torch.backends.cudnn.deterministic = True
        #         # nn.init.kaiming_normal_(m)
        #         nn.init.constant_(m, 0.5)
        #     else:
        #         nn.init.constant_(m, 0)

        for m in self.conv_pred.parameters():
            if m.dim() <= 1:
                nn.init.constant_(m, 0)

        for p in self.conv_pred.shared_convs.parameters():
            if p.dim() > 1:
                torch.manual_seed(0)
                torch.cuda.manual_seed_all(0)
                torch.backends.cudnn.deterministic = True
                nn.init.kaiming_normal_(p)
                # nn.init.constant_(p, 0.5)

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        nn.init.kaiming_normal_(self.conv_pred.conv_cls.weight[0:1])
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        nn.init.kaiming_normal_(self.conv_pred.conv_cls.weight[1:19])
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        nn.init.kaiming_normal_(self.conv_pred.conv_reg.weight[0:3])
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        nn.init.kaiming_normal_(self.conv_pred.conv_reg.weight[3:4])
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        nn.init.kaiming_normal_(self.conv_pred.conv_reg.weight[4:5])
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        nn.init.kaiming_normal_(self.conv_pred.conv_reg.weight[5:23])
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        nn.init.kaiming_normal_(self.conv_pred.conv_reg.weight[23:77])

        # print('transformer: ',
        #       self.decoder_layers[1].attentions[0].attn.in_proj_weight)
        # print(self.decoder_layers[1].attentions[0].attn.in_proj_weight.shape)

        # print('self_posembed: ',
        #       self.decoder_self_posembeds[0].position_embedding_head[0].weight)
        # print(self.decoder_self_posembeds[0].position_embedding_head[0].weight.
        #       shape)

        # print('self.decoder_key_proj: ', self.decoder_key_proj.weight)
        # print(self.decoder_key_proj.weight.shape)

        # print('conv_cls_obj: ', self.conv_pred.conv_cls.weight[0])
        # print(self.conv_pred.conv_cls.weight[0:1].shape)
        # print('conv_res_obj: ', self.conv_pred.conv_reg.weight[23:23+54])
        # print(self.conv_pred.conv_reg.weight[23:23+54].shape)

    def _get_cls_out_channels(self):
        """Return the channel number of classification outputs."""
        # Class numbers (k) + objectness (1)
        return self.num_classes + 1

    def _get_reg_out_channels(self):
        """Return the channel number of regression outputs."""
        # center residual (3),
        # heading class+residual (num_dir_bins*2),
        # size class+residual(num_sizes*4 or 3)
        if self.size_cls_agnostic:
            return 6 + self.num_dir_bins * 2
        else:
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

        seed_xyz, seed_features, seed_indices = self._extract_input(feat_dict)
        print('seed_features: ', seed_features)
        print(seed_features.shape)
        results = dict(
            seed_points=seed_xyz,
            seed_features=seed_features,
            seed_indices=seed_indices)

        # 1. Initial object candidates sampling.
        if sample_mod == 'fps':
            sample_inds = self.fps_module(seed_xyz, seed_features)
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

        candidate_xyz, candidate_features, sample_inds = self.gsample_module(
            seed_xyz, seed_features, sample_inds)

        results['query_points_xyz'] = candidate_xyz  # (B, M, 3)
        results['query_points_feature'] = candidate_features  # (B, C, M)
        results['query_points_sample_inds'] = sample_inds.long()  # (B, M)

        suffix = '_proposal'
        cls_predictions, reg_predictions = self.conv_pred(candidate_features)
        decode_res = self.bbox_coder.split_pred(cls_predictions,
                                                reg_predictions, candidate_xyz,
                                                suffix)

        results.update(decode_res)
        bbox3d = self.bbox_coder.decode(results, suffix)

        # 2. Iterative object box prediction by transformer decoder.
        base_bbox3d = bbox3d[:, :, :6].detach().clone()

        # print('proposal_center: ', bbox3d[:, :, :3].detach().clone())
        # print(bbox3d[:, :, :3].detach().clone().shape)
        # print('proposal_size: ', bbox3d[:, :, 3:6].detach().clone())
        # print(bbox3d[:, :, 3:6].detach().clone().shape)

        query = self.decoder_query_proj(candidate_features).permute(2, 0, 1)
        key = self.decoder_key_proj(seed_features).permute(2, 0, 1)
        value = key

        print('q: ', query)
        print(query.shape)
        print('k: ', key)
        print(key.shape)

        # transformer decoder
        results['num_decoder_layers'] = 0
        for i in range(self.num_decoder_layers):
            suffix = f'_{i}'

            query_pos = self.decoder_self_posembeds[i](base_bbox3d).permute(
                2, 0, 1)
            if i == 0:
                print('base_bbox3d: ', base_bbox3d)
                print(base_bbox3d.shape)
                # print(
                #     'conv1: ', self.decoder_self_posembeds[i].
                #     position_embedding_head[0].weight)
                # print(
                #     'conv1 bias: ', self.decoder_self_posembeds[i].
                #     position_embedding_head[0].bias.sum())
                # print(
                #     'conv2: ', self.decoder_self_posembeds[i].
                #     position_embedding_head[3].weight)
                # print(
                #     'conv2 bias: ', self.decoder_self_posembeds[i].
                #     position_embedding_head[3].bias.sum())
                print('query_pos: ', query_pos)
                print(query_pos.shape)
                print(query_pos.sum())
            key_pos = self.decoder_cross_posembeds[i](seed_xyz).permute(
                2, 0, 1)

            query = self.decoder_layers[i](
                query, key, value, query_pos=query_pos,
                key_pos=key_pos).permute(1, 2, 0)

            results['query' + suffix] = query

            cls_predictions, reg_predictions = self.prediction_heads[i](query)
            decode_res = self.bbox_coder.split_pred(cls_predictions,
                                                    reg_predictions,
                                                    candidate_xyz, suffix)
            # TODO: should save bbox3d instead of decode_res?
            results.update(decode_res)

            bbox3d = self.bbox_coder.decode(results, suffix)
            results['bbox3d' + suffix] = bbox3d
            base_bbox3d = bbox3d[:, :, :6].detach().clone()
            query = query.permute(2, 0, 1)

            results['num_decoder_layers'] += 1
        print('seeds_obj_cls_logits: ', results['seeds_obj_cls_logits'])
        print(results['seeds_obj_cls_logits'].shape)
        print('query_points_feature: ', results['query_points_feature'])
        print(results['query_points_feature'].shape)

        # print('sem_scores_proposal: ', \
        # results['sem_scores_proposal'][0][:20])
        # print(results['sem_scores_proposal'].sum())
        # print(results['sem_scores_proposal'].shape)

        # print('size_res_norm: ', results['size_res_norm_proposal'])
        # print(results['size_res_norm_proposal'].shape)
        # print(results['size_res_norm_proposal'].sum())

        # print('obj_scores_5: ', results['obj_scores_5'])
        # print(results['obj_scores_5'].shape)

        print('center_5: ', results['center_5'])

        print('query: ', results['query_0'])
        print(results['query_0'].shape)
        return results

    @force_fp32(apply_to=('bbox_preds', ))
    def loss(self,
             bbox_preds,
             points,
             gt_bboxes_3d,
             gt_labels_3d,
             pts_semantic_mask=None,
             pts_instance_mask=None,
             img_metas=None,
             gt_bboxes_ignore=None,
             ret_target=False):
        """Compute loss.

        Args:
            bbox_preds (dict): Predictions from forward of vote head.
            points (list[torch.Tensor]): Input points.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth \
                bboxes of each sample.
            gt_labels_3d (list[torch.Tensor]): Labels of each sample.
            pts_semantic_mask (None | list[torch.Tensor]): Point-wise
                semantic mask.
            pts_instance_mask (None | list[torch.Tensor]): Point-wise
                instance mask.
            img_metas (list[dict]): Contain pcd and img's meta info.
            gt_bboxes_ignore (None | list[torch.Tensor]): Specify
                which bounding.
            ret_target (Bool): Return targets or not.

        Returns:
            dict: Losses of Votenet.
        """
        targets = self.get_targets(points, gt_bboxes_3d, gt_labels_3d,
                                   pts_semantic_mask, pts_instance_mask,
                                   bbox_preds)
        (sampling_targets, sampling_weights, assigned_size_targets,
         size_class_targets, size_res_targets, dir_class_targets,
         dir_res_targets, center_targets, assigned_center_targets,
         mask_targets, valid_gt_masks, objectness_targets, objectness_weights,
         box_loss_weights, valid_gt_weights) = targets

        # print('assigned_center_targets: ', assigned_center_targets)
        # print(assigned_center_targets.shape)

        batch_size, proposal_num = size_class_targets.shape[:2]

        # calculate objectness classification loss
        sampling_obj_score = bbox_preds['seeds_obj_cls_logits'].reshape(-1, 1)
        sampling_objectness_loss = self.sampling_objectness_loss(
            sampling_obj_score,
            1 - sampling_targets.reshape(-1),
            sampling_weights.reshape(-1),
            avg_factor=batch_size)

        # print('sampling_targets: ', sampling_targets)
        # print(sampling_targets.shape)
        # print(sampling_targets.sum())
        # print('sampling_weights: ', sampling_weights)
        # print(sampling_weights.shape)
        # print(sampling_weights.sum())
        print('sampling_objectness_loss: ', sampling_objectness_loss)

        objectness_loss_sum = 0.0
        semantic_loss_sum = 0.0
        center_loss_sum = 0.0
        dir_class_loss_sum = 0.0
        dir_res_loss_sum = 0.0
        if self.size_cls_agnostic:
            size_reg_loss_sum = 0.0
        else:
            size_class_loss_sum = 0.0
            size_res_loss_sum = 0.0

        suffixes = ['_proposal'] + [
            f'_{i}' for i in range(bbox_preds['num_decoder_layers'])
        ]
        for suffix in suffixes:

            # calculate objectness loss
            obj_score = bbox_preds['obj_scores' + suffix].transpose(2,
                                                                    1).reshape(
                                                                        -1, 1)
            objectness_loss = self.objectness_loss(
                obj_score,
                1 - objectness_targets.reshape(-1),
                objectness_weights.reshape(-1),
                avg_factor=batch_size)

            objectness_loss_sum += objectness_loss

            print(f'objectness_loss{suffix}: ', objectness_loss)

            # print('objectness_targets: ', objectness_targets)
            # print(objectness_targets.shape)
            # print('objectness_weights: ', objectness_weights)
            # print(objectness_weights.shape)
            # print('obj_score: ', obj_score)
            # print(obj_score.shape)
            # print('objectness_loss: ', objectness_loss)

            # calculate center loss
            box_loss_weights_expand = box_loss_weights.unsqueeze(-1).repeat(
                1, 1, 3)
            center_loss = self.center_loss(
                bbox_preds['center' + suffix],
                assigned_center_targets,
                weight=box_loss_weights_expand)

            center_loss_sum += center_loss
            print(f'center_loss{suffix}: ', center_loss)
            # print(box_loss_weights_expand)
            # print(bbox_preds['center' + suffix])
            # print(bbox_preds['center' + suffix].shape)
            # print("center_loss: ", center_loss)

            # calculate direction class loss
            # print('box_loss_weights: ', box_loss_weights)
            # print(box_loss_weights.shape)
            dir_class_loss = self.dir_class_loss(
                bbox_preds['dir_class' + suffix].transpose(2, 1),
                dir_class_targets,
                weight=box_loss_weights)

            dir_class_loss_sum += dir_class_loss
            print(f'dir_class_loss{suffix}: ', dir_class_loss)
            # print(bbox_preds['dir_class_5'])
            # print("dir_class_targets: ", dir_class_targets)
            # print(dir_class_targets.shape)
            # print("dir_class_loss: ", dir_class_loss)

            # calculate direction residual loss
            heading_label_one_hot = size_class_targets.new_zeros(
                (batch_size, proposal_num, self.num_dir_bins))
            heading_label_one_hot.scatter_(2, dir_class_targets.unsqueeze(-1),
                                           1)
            dir_res_norm = torch.sum(
                bbox_preds['dir_res_norm' + suffix] * heading_label_one_hot,
                -1)
            dir_res_loss = self.dir_res_loss(
                dir_res_norm, dir_res_targets, weight=box_loss_weights)

            dir_res_loss_sum += dir_res_loss
            print(f'dir_res_loss{suffix}: ', dir_res_loss)

            # print('dir_res_loss: ', dir_res_loss)

            if self.size_cls_agnostic:
                size_reg_loss = self.size_reg_loss(
                    bbox_preds['size' + suffix],
                    assigned_size_targets,
                    weight=box_loss_weights_expand)

                size_reg_loss_sum += size_reg_loss
                # if suffix == '_proposal':
                #     # print('pred_size: ', bbox_preds['size' + suffix])
                #     # print(bbox_preds['size' + suffix].shape)
                #     print('assigned_size_targets: ', assigned_size_targets)
                #     print(assigned_size_targets.shape)
                print(f'size_reg_loss{suffix}: ', size_reg_loss)
            else:
                # calculate size class loss
                size_class_loss = self.size_class_loss(
                    bbox_preds['size_class' + suffix].transpose(2, 1),
                    size_class_targets,
                    weight=box_loss_weights)

                size_class_loss_sum += size_class_loss

                print(f'size_class_loss{suffix}: ', size_class_loss)

                # calculate size residual loss
                one_hot_size_targets = size_class_targets.new_zeros(
                    (batch_size, proposal_num, self.num_sizes))
                one_hot_size_targets.scatter_(2,
                                              size_class_targets.unsqueeze(-1),
                                              1)
                one_hot_size_targets_expand = one_hot_size_targets.unsqueeze(
                    -1).repeat(1, 1, 1, 3).contiguous()
                size_residual_norm = torch.sum(
                    bbox_preds['size_res_norm' + suffix] *
                    one_hot_size_targets_expand, 2)
                box_loss_weights_expand = box_loss_weights.unsqueeze(
                    -1).repeat(1, 1, 3)
                size_res_loss = self.size_res_loss(
                    size_residual_norm,
                    size_res_targets,
                    weight=box_loss_weights_expand)

                size_res_loss_sum += size_res_loss

                # print('size_residual_norm: ', size_residual_norm)
                # print('box_loss_weights_expand: ', box_loss_weights_expand)
                print(f'size_res_loss{suffix}: ', size_res_loss)

                # print(size_res_targets)
                # print(size_res_targets.shape)
                # print("size_res_loss: ", size_res_loss)

            # calculate semantic loss
            semantic_loss = self.semantic_loss(
                bbox_preds['sem_scores' + suffix].transpose(2, 1),
                mask_targets,
                weight=box_loss_weights)

            semantic_loss_sum += semantic_loss

            print(f'semantic_loss{suffix}: ', semantic_loss)

            # print('semantic_loss: ', semantic_loss)

        print('query_points_generation_loss: ', sampling_objectness_loss / 8)

        if self.size_cls_agnostic:
            sum_heads_box_loss = center_loss_sum + dir_class_loss_sum + \
                dir_res_loss_sum + size_reg_loss_sum
        else:
            sum_heads_box_loss = center_loss_sum + dir_class_loss_sum + \
                dir_res_loss_sum + size_class_loss_sum + size_res_loss_sum
        print('sum_heads_box_loss: ', sum_heads_box_loss)
        print('sum_heads_objectness_loss: ', objectness_loss_sum)
        print('sum_heads_sem_cls_loss: ', semantic_loss_sum)

        loss_sum = sampling_objectness_loss + \
            (objectness_loss_sum + sum_heads_box_loss + semantic_loss_sum) / 7
        print('loss: ', loss_sum)

        objectness_loss_sum /= len(suffixes)
        semantic_loss_sum /= len(suffixes)
        center_loss_sum /= len(suffixes)
        dir_class_loss_sum /= len(suffixes)
        dir_res_loss_sum /= len(suffixes)

        losses = dict(
            sampling_objectness_loss=sampling_objectness_loss,
            objectness_loss=objectness_loss_sum,
            semantic_loss=semantic_loss_sum,
            center_loss=center_loss_sum,
            dir_class_loss=dir_class_loss_sum,
            dir_res_loss=dir_res_loss_sum)

        if self.size_cls_agnostic:
            size_reg_loss_sum /= len(suffixes)
            losses['size_reg_loss'] = size_reg_loss_sum
        else:
            size_class_loss_sum /= len(suffixes)
            size_res_loss_sum /= len(suffixes)
            losses['size_class_loss'] = size_class_loss_sum
            losses['size_res_loss'] = size_res_loss_sum

        # if self.iou_loss:
        #     corners_pred = self.bbox_coder.decode_corners(
        #         bbox_preds['center'], size_residual_norm,
        #         one_hot_size_targets_expand)
        #     corners_target = self.bbox_coder.decode_corners(
        #         assigned_center_targets, size_res_targets,
        #         one_hot_size_targets_expand)
        #     iou_loss = self.iou_loss(
        #         corners_pred, corners_target, weight=box_loss_weights)
        #     losses['iou_loss'] = iou_loss

        if ret_target:
            losses['targets'] = targets

        return losses

    def get_targets(self,
                    points,
                    gt_bboxes_3d,
                    gt_labels_3d,
                    pts_semantic_mask=None,
                    pts_instance_mask=None,
                    bbox_preds=None):
        """Generate targets of vote head.

        Args:
            points (list[torch.Tensor]): Points of each batch.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth \
                bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): Labels of each batch.
            pts_semantic_mask (None | list[torch.Tensor]): Point-wise semantic
                label of each batch.
            pts_instance_mask (None | list[torch.Tensor]): Point-wise instance
                label of each batch.
            bbox_preds (torch.Tensor): Bounding box predictions of vote head.

        Returns:
            tuple[torch.Tensor]: Targets of vote head.
        """
        # find empty example
        valid_gt_masks = list()
        gt_num = list()
        for index in range(len(gt_labels_3d)):
            if len(gt_labels_3d[index]) == 0:
                fake_box = gt_bboxes_3d[index].tensor.new_zeros(
                    1, gt_bboxes_3d[index].tensor.shape[-1])
                gt_bboxes_3d[index] = gt_bboxes_3d[index].new_box(fake_box)
                gt_labels_3d[index] = gt_labels_3d[index].new_zeros(1)
                valid_gt_masks.append(gt_labels_3d[index].new_zeros(1))
                gt_num.append(1)
            else:
                valid_gt_masks.append(gt_labels_3d[index].new_ones(
                    gt_labels_3d[index].shape))
                gt_num.append(gt_labels_3d[index].shape[0])
        max_gt_num = max(gt_num)

        max_gt_nums = [max_gt_num for _ in range(len(gt_labels_3d))]

        if pts_semantic_mask is None:
            pts_semantic_mask = [None for i in range(len(gt_labels_3d))]
            pts_instance_mask = [None for i in range(len(gt_labels_3d))]

        seed_points = [
            bbox_preds['seed_points'][i] for i in range(len(gt_labels_3d))
        ]

        seed_indices = [
            bbox_preds['seed_indices'][i] for i in range(len(gt_labels_3d))
        ]

        candidate_indices = [
            bbox_preds['query_points_sample_inds'][i]
            for i in range(len(gt_labels_3d))
        ]

        (sampling_targets, assigned_size_targets, size_class_targets,
         size_res_targets, dir_class_targets, dir_res_targets, center_targets,
         assigned_center_targets, mask_targets, objectness_targets,
         objectness_masks) = multi_apply(self.get_targets_single, points,
                                         gt_bboxes_3d, gt_labels_3d,
                                         pts_semantic_mask, pts_instance_mask,
                                         max_gt_nums, seed_points,
                                         seed_indices, candidate_indices)

        # pad targets as original code of GroupFree3D.
        for index in range(len(gt_labels_3d)):
            # +1: the last one for the empty target ?
            pad_num = max_gt_num - gt_labels_3d[index].shape[0] + 1
            # center_targets[index] = F.pad(center_targets[index],
            #                               (0, 0, 0, pad_num))
            valid_gt_masks[index] = F.pad(valid_gt_masks[index], (0, pad_num))

        sampling_targets = torch.stack(sampling_targets)
        sampling_weights = (sampling_targets >= 0).float()
        sampling_normalizer = sampling_weights.sum(dim=1, keepdim=True).float()
        sampling_weights /= torch.clamp(sampling_normalizer, min=1.0)

        assigned_size_targets = torch.stack(assigned_size_targets)
        center_targets = torch.stack(center_targets)
        valid_gt_masks = torch.stack(valid_gt_masks)

        assigned_center_targets = torch.stack(assigned_center_targets)
        objectness_targets = torch.stack(objectness_targets)

        objectness_weights = torch.stack(objectness_masks)
        # objectness_weights /= (torch.sum(objectness_weights) + 1e-6)
        cls_normalizer = objectness_weights.sum(dim=1, keepdim=True).float()
        objectness_weights /= torch.clamp(cls_normalizer, min=1.0)

        box_loss_weights = objectness_targets.float() / (
            torch.sum(objectness_targets).float() + 1e-6)

        valid_gt_weights = valid_gt_masks.float() / (
            torch.sum(valid_gt_masks.float()) + 1e-6)

        dir_class_targets = torch.stack(dir_class_targets)
        dir_res_targets = torch.stack(dir_res_targets)
        size_class_targets = torch.stack(size_class_targets)
        size_res_targets = torch.stack(size_res_targets)
        mask_targets = torch.stack(mask_targets)

        return (sampling_targets, sampling_weights, assigned_size_targets,
                size_class_targets, size_res_targets, dir_class_targets,
                dir_res_targets, center_targets, assigned_center_targets,
                mask_targets, valid_gt_masks, objectness_targets,
                objectness_weights, box_loss_weights, valid_gt_weights)

    def get_targets_single(self,
                           points,
                           gt_bboxes_3d,
                           gt_labels_3d,
                           pts_semantic_mask=None,
                           pts_instance_mask=None,
                           max_gt_nums=None,
                           seed_points=None,
                           seed_indices=None,
                           candidate_indices=None,
                           seed_points_obj_topk=4):
        """Generate targets of vote head for single batch.

        Args:
            points (torch.Tensor): Points of each batch.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): Ground truth \
                boxes of each batch.
            gt_labels_3d (torch.Tensor): Labels of each batch.
            pts_semantic_mask (None | torch.Tensor): Point-wise semantic
                label of each batch.
            pts_instance_mask (None | torch.Tensor): Point-wise instance
                label of each batch.
            aggregated_points (torch.Tensor): Aggregated points from
                vote aggregation layer.

        Returns:
            tuple[torch.Tensor]: Targets of vote head.
        """
        assert self.bbox_coder.with_rot or pts_semantic_mask is not None

        gt_bboxes_3d = gt_bboxes_3d.to(points.device)

        # 0. generate pts_instance_label and pts_obj_mask
        num_points = points.shape[0]
        pts_obj_mask = points.new_zeros([num_points], dtype=torch.long)
        pts_instance_label = points.new_zeros([num_points],
                                              dtype=torch.long) - 1
        # print('pts_instance_mask: ', pts_instance_mask)
        # print(pts_instance_mask.shape)
        # print(pts_instance_mask.sum())
        # print(pts_instance_mask[:50])
        # print('pts_semantic_mask: ', pts_semantic_mask)
        # print(pts_semantic_mask.shape)
        # print(pts_semantic_mask.sum())
        # print(pts_semantic_mask[:50])
        # print('gt_bboxes_3d.gravity_center: ', gt_bboxes_3d.gravity_center)
        # print(gt_bboxes_3d.gravity_center.shape)
        if self.bbox_coder.with_rot:
            vote_targets = points.new_zeros([num_points, 4 * self.gt_per_seed])
            vote_target_idx = points.new_zeros([num_points], dtype=torch.long)
            box_indices_all = gt_bboxes_3d.points_in_boxes(points)  # (N, M)
            for i in range(gt_labels_3d.shape[0]):
                box_indices = box_indices_all[:, i]  # (N)
                indices = torch.nonzero(
                    box_indices, as_tuple=False).squeeze(-1)  # (K)
                selected_points = points[indices]
                pts_obj_mask[indices] = 1
                vote_targets_tmp = vote_targets[indices]  # (K, 12)
                votes = gt_bboxes_3d.gravity_center[i].unsqueeze(
                    0) - selected_points[:, :3]  # (K, 3)

                for j in range(self.gt_per_seed):
                    # (Q) : Q <= K
                    column_indices = torch.nonzero(
                        vote_target_idx[indices] == j,
                        as_tuple=False).squeeze(-1)
                    vote_targets_tmp[column_indices,
                                     int(j * 3):int(j * 3 +
                                                    3)] = votes[column_indices]

                    vote_targets_tmp[column_indices,
                                     j + 3 * self.gt_per_seed] = i
                    if j == 0:
                        vote_targets_tmp[
                            column_indices, :3 *
                            self.gt_per_seed] = votes[column_indices].repeat(
                                1, self.gt_per_seed)

                        vote_targets_tmp[column_indices,
                                         3 * self.gt_per_seed:] = i

                vote_targets[indices] = vote_targets_tmp
                vote_target_idx[indices] = torch.clamp(
                    vote_target_idx[indices] + 1, max=2)

            dist = points.new_zeros([num_points, self.gt_per_seed]) + 1000
            for j in range(self.gt_per_seed):
                dist[:, j] = (vote_targets[:, 3 * j:3 * j + 3]**2).sum(-1)
            instance_indices = torch.argmin(
                dist, dim=-1).unsqueeze(-1) + 3 * self.gt_per_seed
            instance_lable = torch.gather(vote_targets, 1,
                                          instance_indices).squeeze(-1)
            pts_instance_label = instance_lable.long()
            pts_instance_label[pts_obj_mask == 0] = -1

        elif pts_semantic_mask is not None:
            for i in torch.unique(pts_instance_mask):
                indices = torch.nonzero(
                    pts_instance_mask == i, as_tuple=False).squeeze(-1)

                if pts_semantic_mask[indices[0]] < self.num_classes:
                    selected_points = points[indices, :3]
                    center = 0.5 * (
                        selected_points.min(0)[0] + selected_points.max(0)[0])

                    delta_xyz = center - gt_bboxes_3d.gravity_center
                    instance_lable = torch.argmin((delta_xyz**2).sum(-1))
                    pts_instance_label[indices] = instance_lable
                    pts_obj_mask[indices] = 1
        else:
            raise NotImplementedError

        # pts_instance_label = pts_instance_mask
        # pts_obj_mask = pts_semantic_mask

        print('pts_obj_mask: ', pts_obj_mask)
        print(pts_obj_mask.shape)
        print(pts_obj_mask.sum())
        print(pts_obj_mask[:50])

        print('pts_instance_label: ', pts_instance_label)
        print(pts_instance_label.shape)
        print(pts_instance_label.sum())
        print(pts_instance_label[:50])

        # generate center, dir, size target
        (center_targets, size_targets, size_class_targets, size_res_targets,
         dir_class_targets,
         dir_res_targets) = self.bbox_coder.encode(gt_bboxes_3d, gt_labels_3d)

        # pad targets as original code of GroupFree3D
        pad_num = max_gt_nums - gt_labels_3d.shape[0] + 1
        gt_labels_3d = F.pad(gt_labels_3d, (0, pad_num))
        gt_center_pad = F.pad(
            gt_bboxes_3d.tensor, (0, 0, 0, pad_num), value=1000)
        gt_bboxes_3d = gt_bboxes_3d.new_box(gt_center_pad)
        center_targets = F.pad(center_targets, (0, 0, 0, pad_num))
        size_targets = F.pad(size_targets, (0, 0, 0, pad_num))
        print('size_targets: ', size_targets)
        print(size_targets.shape)
        size_class_targets = F.pad(size_class_targets, (0, pad_num))
        size_res_targets = F.pad(size_res_targets, (0, 0, 0, pad_num))
        dir_class_targets = F.pad(dir_class_targets, (0, pad_num))
        dir_res_targets = F.pad(dir_res_targets, (0, pad_num))

        # 1. generate objectness targets in sampling head
        gt_num = gt_labels_3d.shape[0]
        num_seed = seed_points.shape[0]
        num_candidate = candidate_indices.shape[0]

        # pts_instance_label
        # num_seed
        object_assignment = torch.gather(pts_instance_label, 0, seed_indices)
        # set background points to the last gt bbox ??
        object_assignment[object_assignment < 0] = gt_num - 1
        object_assignment_one_hot = gt_bboxes_3d.tensor.new_zeros(
            (num_seed, gt_num))
        object_assignment_one_hot.scatter_(1, object_assignment.unsqueeze(-1),
                                           1)  # (num_seed, gt_num)

        delta_xyz = seed_points.unsqueeze(
            1) - gt_bboxes_3d.gravity_center.unsqueeze(
                0)  # (num_seed, gt_num, 3)
        delta_xyz = delta_xyz / (gt_bboxes_3d.dims.unsqueeze(0) + 1e-6)

        new_dist = torch.sum(delta_xyz**2, dim=-1)
        euclidean_dist1 = torch.sqrt(new_dist + 1e-6)
        euclidean_dist1 = euclidean_dist1 * object_assignment_one_hot + 100 * (
            1 - object_assignment_one_hot)
        # (gt_num, num_seed)
        euclidean_dist1 = euclidean_dist1.transpose(0, 1).contiguous()

        topk_inds = torch.topk(
            euclidean_dist1, seed_points_obj_topk,
            largest=False)[1]  # gt_numxtopk
        topk_inds = topk_inds.long()
        topk_inds = topk_inds.view(-1).contiguous()

        sampling_targets = torch.zeros(
            num_seed + 1, dtype=torch.long).to(points.device)
        sampling_targets[topk_inds] = 1
        sampling_targets = sampling_targets[:num_seed]
        # pts_instance_label
        objectness_label_mask = torch.gather(pts_instance_label, 0,
                                             seed_indices)  # num_seed
        sampling_targets[objectness_label_mask < 0] = 0

        # 2. objectness target
        # pts_obj_mask
        seed_obj_gt = torch.gather(pts_obj_mask, 0, seed_indices)  # num_seed
        objectness_targets = torch.gather(seed_obj_gt, 0,
                                          candidate_indices)  # query_points

        # objectness_targets[objectness_targets < self.num_classes] = 1
        # objectness_targets[objectness_targets == self.num_classes] = 0

        # print('objectness_targets: ', objectness_targets)
        # print(objectness_targets.shape)

        # 3. box target
        # pts_instance_label
        seed_instance_label = torch.gather(pts_instance_label, 0,
                                           seed_indices)  # num_seed
        query_points_instance_label = torch.gather(
            seed_instance_label, 0, candidate_indices)  # num_candidate

        # Set assignment
        # (num_candidate, ) with values in 0,1,...,gt_num-1
        assignment = query_points_instance_label
        assignment[assignment <
                   0] = gt_num - 1  # set background points to the last gt bbox

        # print(size_res_targets)
        # print(size_res_targets.shape)

        assigned_center_targets = center_targets[assignment]
        assignment_expand = assignment.unsqueeze(1).repeat(1, 3)
        # assigned_center_targets = torch.gather(center_targets, 0,
        #                                        assignment_expand)

        # print(assignment_expand)
        assigned_size_targets = size_targets[assignment]
        dir_class_targets = dir_class_targets[assignment]
        # dir_class_targets = torch.gather(dir_class_targets, 0, assignment)

        dir_res_targets = dir_res_targets[assignment]
        # dir_res_targets = torch.gather(dir_res_targets, 0, assignment)
        dir_res_targets /= (np.pi / self.num_dir_bins)

        size_class_targets = size_class_targets[assignment]
        # size_class_targets = torch.gather(size_class_targets, 0, assignment)
        # print('size res targets: ', size_res_targets)
        # print(size_res_targets.shape)
        # print('object_assignment: ', assignment)
        # print(assignment.shape)
        # size_res_targets = size_res_targets[assignment]

        size_res_targets = \
            torch.gather(size_res_targets, 0, assignment_expand)
        # print('size res targets: ', size_res_targets)
        # print(size_res_targets.shape)
        one_hot_size_targets = gt_bboxes_3d.tensor.new_zeros(
            (num_candidate, self.num_sizes))

        one_hot_size_targets.scatter_(1, size_class_targets.unsqueeze(-1), 1)
        one_hot_size_targets = one_hot_size_targets.unsqueeze(-1).repeat(
            1, 1, 3)  # (num_candidate,num_size_cluster,3)

        mean_sizes = size_res_targets.new_tensor(
            self.bbox_coder.mean_sizes).unsqueeze(0)
        pos_mean_sizes = torch.sum(one_hot_size_targets * mean_sizes, 1)
        size_res_targets /= pos_mean_sizes
        # print('size res targets: ', size_res_targets)
        # print(size_res_targets.shape)

        mask_targets = gt_labels_3d[assignment]
        # mask_targets = torch.gather(gt_labels_3d, 0, assignment)

        objectness_masks = points.new_ones((num_candidate))

        return (sampling_targets, assigned_size_targets, size_class_targets,
                size_res_targets, dir_class_targets,
                dir_res_targets, center_targets, assigned_center_targets,
                mask_targets.long(), objectness_targets, objectness_masks)

    def get_bboxes(self,
                   points,
                   bbox_preds,
                   input_metas,
                   rescale=False,
                   use_nms=True):
        """Generate bboxes from vote head predictions.

        Args:
            points (torch.Tensor): Input points.
            bbox_preds (dict): Predictions from vote head.
            input_metas (list[dict]): Point cloud and image's meta info.
            rescale (bool): Whether to rescale bboxes.
            use_nms (bool): Whether to apply NMS, skip nms postprocessing
                while using vote head in rpn stage.

        Returns:
            list[tuple[torch.Tensor]]: Bounding boxes, scores and labels.
        """
        # for suffix in self.test_cfg['suffixes']:
        suffix = self.test_cfg['suffixes']

        # decode boxes
        obj_scores = F.sigmoid(bbox_preds['obj_scores' + suffix])[..., -1]
        sem_scores = F.softmax(bbox_preds['sem_scores' + suffix], dim=-1)
        bbox3d = self.bbox_coder.decode(bbox_preds, suffix)
        if use_nms:
            batch_size = bbox3d.shape[0]
            results = list()
            for b in range(batch_size):
                bbox_selected, score_selected, labels = \
                    self.multiclass_nms_single(obj_scores[b], sem_scores[b],
                                               bbox3d[b], points[b, ..., :3],
                                               input_metas[b])
                bbox = input_metas[b]['box_type_3d'](
                    bbox_selected,
                    box_dim=bbox_selected.shape[-1],
                    with_yaw=self.bbox_coder.with_rot)
                results.append((bbox, score_selected, labels))

            return results
        else:
            return bbox3d

    def multiclass_nms_single(self, obj_scores, sem_scores, bbox, points,
                              input_meta):
        """Multi-class nms in single batch.

        Args:
            obj_scores (torch.Tensor): Objectness score of bounding boxes.
            sem_scores (torch.Tensor): semantic class score of bounding boxes.
            bbox (torch.Tensor): Predicted bounding boxes.
            points (torch.Tensor): Input points.
            input_meta (dict): Point cloud and image's meta info.

        Returns:
            tuple[torch.Tensor]: Bounding boxes, scores and labels.
        """
        bbox = input_meta['box_type_3d'](
            bbox,
            box_dim=bbox.shape[-1],
            with_yaw=self.bbox_coder.with_rot,
            origin=(0.5, 0.5, 0.5))
        box_indices = bbox.points_in_boxes(points)

        corner3d = bbox.corners
        minmax_box3d = corner3d.new(torch.Size((corner3d.shape[0], 6)))
        minmax_box3d[:, :3] = torch.min(corner3d, dim=1)[0]
        minmax_box3d[:, 3:] = torch.max(corner3d, dim=1)[0]

        nonempty_box_mask = box_indices.T.sum(1) > 5

        bbox_classes = torch.argmax(sem_scores, -1)
        nms_selected = aligned_3d_nms(minmax_box3d[nonempty_box_mask],
                                      obj_scores[nonempty_box_mask],
                                      bbox_classes[nonempty_box_mask],
                                      self.test_cfg.nms_thr)

        # filter empty boxes and boxes with low score
        scores_mask = (obj_scores > self.test_cfg.score_thr)
        nonempty_box_inds = torch.nonzero(
            nonempty_box_mask, as_tuple=False).flatten()
        nonempty_mask = torch.zeros_like(bbox_classes).scatter(
            0, nonempty_box_inds[nms_selected], 1)
        selected = (nonempty_mask.bool() & scores_mask.bool())

        if self.test_cfg.per_class_proposal:
            bbox_selected, score_selected, labels = [], [], []
            for k in range(sem_scores.shape[-1]):
                bbox_selected.append(bbox[selected].tensor)
                score_selected.append(obj_scores[selected] *
                                      sem_scores[selected][:, k])
                labels.append(
                    torch.zeros_like(bbox_classes[selected]).fill_(k))
            bbox_selected = torch.cat(bbox_selected, 0)
            score_selected = torch.cat(score_selected, 0)
            labels = torch.cat(labels, 0)
        else:
            bbox_selected = bbox[selected].tensor
            score_selected = obj_scores[selected]
            labels = bbox_classes[selected]

        return bbox_selected, score_selected, labels
