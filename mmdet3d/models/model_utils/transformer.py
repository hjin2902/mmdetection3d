from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from torch import nn as nn

from mmdet3d.models.dense_heads.base_conv_bbox_head import BaseConvBboxHead
from mmdet.core import build_bbox_coder


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


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class GroupFree3DTransformerDecoder(TransformerLayerSequence):
    """TransformerDeocder in `Group-Free 3D
    https://arxiv.org/abs/2104.00678>`_.

    As subclass of TransformerLayerSequence.

    Args:
        transformerlayer (list[obj:`mmcv.ConfigDict`] |
            obj:`mmcv.ConfigDict`): Config of transformerlayer
            in TransformerCoder. If it is obj:`mmcv.ConfigDict`,
             it would be repeated `num_layer` times to a
             list[`mmcv.ConfigDict`]. Default: None.
        num_layers (int): The number of `TransformerLayer`. Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self, transformerlayers=None, num_layers=None, init_cfg=None):
        super(GroupFree3DTransformerDecoder, self).__init__(
            transformerlayers=None, num_layers=None, init_cfg=None)
        assert self.init_cfg is not None
        self.bbox_coder = build_bbox_coder(self.init_cfg['bbox_coder'])
        # prediction heads
        self.pred_layer_cfg = self.init_cfg['pred_layer_cfg']
        self.prediction_heads = nn.ModuleList()
        for _ in range(self.num_layers):
            self.prediction_heads.append(
                BaseConvBboxHead(
                    **self.pred_layer_cfg,
                    num_cls_out_channels=self._get_cls_out_channels(),
                    num_reg_out_channels=self._get_reg_out_channels()))

        # query proj and key proj
        self.decoder_query_proj = nn.Conv1d(
            self.embed_dims, self.embed_dims, kernel_size=1)
        self.decoder_key_proj = nn.Conv1d(
            self.embed_dims, self.embed_dims, kernel_size=1)

        # query position embed
        self.decoder_self_posembeds = nn.ModuleList()
        for _ in range(self.num_layers):
            self.decoder_self_posembeds.append(
                PositionEmbeddingLearned(6, self.embed_dims))
        # key position embed
        self.decoder_cross_posembeds = nn.ModuleList()
        for _ in range(self.num_layers):
            self.decoder_cross_posembeds.append(
                PositionEmbeddingLearned(3, self.embed_dims))

    def forward(self,
                candidate_features,
                candidate_xyz,
                seed_features,
                seed_xyz,
                base_bbox3d,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `GroupFree3DTransformerDecoder`.

        Args:
            candidate_features (Tensor): Candidate features after initial
                sampling with shape `(B, C, M)`.
            candidate_xyz (Tensor): The coordinate of candidate features
                with shape `(B, M, 3)`.
            seed_features (Tensor): Seed features from backbone with shape
                `(B, C, N)`.
            seed_xyz (Tensor): The coordinate of seed features with shape
                `(B, N, 3)`.
            base_bbox3d (Tensor): The initial predicted candidates with
                shape `(B, M, 6)`.
            attn_masks (List[Tensor], optional): Each element is 2D Tensor
                which is used in calculation of corresponding attention in
                operation_order. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in self-attention
                Default: None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Dict: predicted 3Dboxes of all layers.
        """
        query = self.decoder_query_proj(candidate_features).permute(2, 0, 1)
        key = self.decoder_key_proj(seed_features).permute(2, 0, 1)
        value = key

        results = {}

        for i in range(self.num_layers):
            suffix = f'_{i}'

            query_pos = self.decoder_self_posembeds[i](base_bbox3d).permute(
                2, 0, 1)
            key_pos = self.decoder_cross_posembeds[i](seed_xyz).permute(
                2, 0, 1)

            query = self.layers[i](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs)

            cls_predictions, reg_predictions = self.prediction_heads[i](query)
            decode_res = self.bbox_coder.split_pred(cls_predictions,
                                                    reg_predictions,
                                                    candidate_xyz, suffix)
            # should save bbox3d instead of decode_res?
            results.update(decode_res)

            bbox3d = self.bbox_coder.decode(results, suffix)

            results['bbox3d' + suffix] = bbox3d

            base_bbox3d = bbox3d[:, :, :6].detach().clone()
            query = query.permute(2, 0, 1)

        return results
