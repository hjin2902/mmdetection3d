import copy
import numpy as np
import pytest
import random
import torch
from os.path import dirname, exists, join

from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.models.builder import build_detector, build_head

# from mmdet.apis import set_random_seed


def _setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _get_config_directory():
    """Find the predefined detector config directory."""
    try:
        # Assume we are running in the source mmdetection3d repo
        repo_dpath = dirname(dirname(dirname(dirname(__file__))))
    except NameError:
        # For IPython development when this __file__ is not defined
        import mmdet3d
        repo_dpath = dirname(dirname(mmdet3d.__file__))
    config_dpath = join(repo_dpath, 'configs')
    if not exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath


def _get_config_module(fname):
    """Load a configuration as a python module."""
    from mmcv import Config
    config_dpath = _get_config_directory()
    config_fpath = join(config_dpath, fname)
    config_mod = Config.fromfile(config_fpath)
    return config_mod


def _get_vote_head_cfg(fname):
    """Grab configs necessary to create a vote_head.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    import mmcv
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    train_cfg = mmcv.Config(copy.deepcopy(config.model.train_cfg))
    test_cfg = mmcv.Config(copy.deepcopy(config.model.test_cfg))

    vote_head = model.bbox_head
    vote_head.update(train_cfg=train_cfg)
    vote_head.update(test_cfg=test_cfg)
    return vote_head


def _get_detector_cfg(fname):
    """Grab configs necessary to create a detector.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    import mmcv
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    train_cfg = mmcv.Config(copy.deepcopy(config.model.train_cfg))
    test_cfg = mmcv.Config(copy.deepcopy(config.model.test_cfg))

    model.update(train_cfg=train_cfg)
    model.update(test_cfg=test_cfg)
    return model


def load_checkpoint_head(checkpoint_path, model):
    # Load checkpoint if there is any

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model']

    for k in list(state_dict.keys()):
        # print(k, '\t', state_dict[k].shape)
        # state_dict[k[len("module."):]] = state_dict[k]
        # delete renamed or unused k
        # del state_dict[k]

        if 'backbone_net' in k:
            del state_dict[k]
            continue

        if 'points_obj_cls' in k:
            state_dict[k[len('module.'):]] = state_dict[k]

        elif 'proposal_head' in k:
            if '1' in k:
                a = '0.'
                b, c = k[len('module.proposal_head.'):].split('1')
                key = 'conv_pred.shared_convs.layer' + a + b + c
            elif '2' in k:
                a = '1.'
                b, c = k[len('module.proposal_head.'):].split('2')
                key = 'conv_pred.shared_convs.layer' + a + b + c
            else:
                key = 'conv_pred.' + k[len('module.proposal_head.'):]
            state_dict[key] = state_dict[k]

        elif 'decoder_key_proj' in k:
            state_dict[k[len('module.'):]] = state_dict[k]

        elif 'decoder_query_proj' in k:
            state_dict[k[len('module.'):]] = state_dict[k]

        elif 'decoder_self_posembeds' in k:
            state_dict[k[len('module.'):]] = state_dict[k]

        elif 'decoder_cross_posembeds' in k:
            state_dict[k[len('module.'):]] = state_dict[k]

        elif 'self_attn' in k:
            a, b = k.split('.self_attn')
            key = 'decoder_layers.' + a[-1] + '.attentions.0.attn' + b
            state_dict[key] = state_dict[k]

        elif 'multihead_attn' in k:
            a, b = k.split('.multihead_attn')
            key = 'decoder_layers.' + a[-1] + '.attentions.1.attn' + b
            state_dict[key] = state_dict[k]

        elif 'linear' in k:
            a, b = k.split('.linear')
            if b[0] == '1':
                c = '0.0'
            else:
                c = '1'
            key = 'decoder_layers.' + a[-1] + '.ffns.0.layers.' + c + b[1:]
            state_dict[key] = state_dict[k]

        elif 'norm' in k:
            a, b = k.split('.norm')
            c = str(int(b[0]) - 1)
            key = 'decoder_layers.' + a[-1] + '.norms.' + c + b[1:]
            state_dict[key] = state_dict[k]

        elif 'prediction_heads' in k:
            a = k[len('module.prediction_heads.'):]
            b = a[0:2]
            c = a[2:]
            if '1' in c:
                d, e = c.split('1')
                key = 'prediction_heads.' + b + 'shared_convs.layer0.' + d + e
            elif '2' in c:
                d, e = c.split('2')
                key = 'prediction_heads.' + b + 'shared_convs.layer1.' + d + e
            else:
                key = 'prediction_heads.' + a

            state_dict[key] = state_dict[k]

        del state_dict[k]

    obj_weight = state_dict['conv_pred.objectness_scores_head.weight']
    cls_weight = state_dict['conv_pred.sem_cls_scores_head.weight']
    state_dict['conv_pred.conv_cls.weight'] = torch.cat(
        [obj_weight, cls_weight])
    del state_dict['conv_pred.objectness_scores_head.weight']
    del state_dict['conv_pred.sem_cls_scores_head.weight']

    obj_bias = state_dict['conv_pred.objectness_scores_head.bias']
    cls_bias = state_dict['conv_pred.sem_cls_scores_head.bias']
    state_dict['conv_pred.conv_cls.bias'] = torch.cat([obj_bias, cls_bias])
    del state_dict['conv_pred.objectness_scores_head.bias']
    del state_dict['conv_pred.sem_cls_scores_head.bias']

    center_weight = state_dict['conv_pred.center_residual_head.weight']
    dir_cls_weight = state_dict['conv_pred.heading_class_head.weight']
    dir_res_weight = state_dict['conv_pred.heading_residual_head.weight']
    size_cls_weight = state_dict['conv_pred.size_class_head.weight']
    size_res_weight = state_dict['conv_pred.size_residual_head.weight']

    state_dict['conv_pred.conv_reg.weight'] = torch.cat([
        center_weight, dir_cls_weight, dir_res_weight, size_cls_weight,
        size_res_weight
    ])
    del state_dict['conv_pred.center_residual_head.weight']
    del state_dict['conv_pred.heading_class_head.weight']
    del state_dict['conv_pred.heading_residual_head.weight']
    del state_dict['conv_pred.size_class_head.weight']
    del state_dict['conv_pred.size_residual_head.weight']

    center_bias = state_dict['conv_pred.center_residual_head.bias']
    dir_cls_bias = state_dict['conv_pred.heading_class_head.bias']
    dir_res_bias = state_dict['conv_pred.heading_residual_head.bias']
    size_cls_bias = state_dict['conv_pred.size_class_head.bias']
    size_res_bias = state_dict['conv_pred.size_residual_head.bias']

    state_dict['conv_pred.conv_reg.bias'] = torch.cat([
        center_bias, dir_cls_bias, dir_res_bias, size_cls_bias, size_res_bias
    ])
    del state_dict['conv_pred.center_residual_head.bias']
    del state_dict['conv_pred.heading_class_head.bias']
    del state_dict['conv_pred.heading_residual_head.bias']
    del state_dict['conv_pred.size_class_head.bias']
    del state_dict['conv_pred.size_residual_head.bias']

    for i in range(6):
        prefix = 'prediction_heads.' + str(i) + '.'

        obj_weight = state_dict[prefix + 'objectness_scores_head.weight']
        cls_weight = state_dict[prefix + 'sem_cls_scores_head.weight']
        state_dict[prefix + 'conv_cls.weight'] = torch.cat(
            [obj_weight, cls_weight])
        del state_dict[prefix + 'objectness_scores_head.weight']
        del state_dict[prefix + 'sem_cls_scores_head.weight']

        obj_bias = state_dict[prefix + 'objectness_scores_head.bias']
        cls_bias = state_dict[prefix + 'sem_cls_scores_head.bias']
        state_dict[prefix + 'conv_cls.bias'] = torch.cat([obj_bias, cls_bias])
        del state_dict[prefix + 'objectness_scores_head.bias']
        del state_dict[prefix + 'sem_cls_scores_head.bias']

        center_weight = state_dict[prefix + 'center_residual_head.weight']
        dir_cls_weight = state_dict[prefix + 'heading_class_head.weight']
        dir_res_weight = state_dict[prefix + 'heading_residual_head.weight']
        size_cls_weight = state_dict[prefix + 'size_class_head.weight']
        size_res_weight = state_dict[prefix + 'size_residual_head.weight']
        state_dict[prefix + 'conv_reg.weight'] = torch.cat([
            center_weight, dir_cls_weight, dir_res_weight, size_cls_weight,
            size_res_weight
        ])
        del state_dict[prefix + 'center_residual_head.weight']
        del state_dict[prefix + 'heading_class_head.weight']
        del state_dict[prefix + 'heading_residual_head.weight']
        del state_dict[prefix + 'size_class_head.weight']
        del state_dict[prefix + 'size_residual_head.weight']

        center_bias = state_dict[prefix + 'center_residual_head.bias']
        dir_cls_bias = state_dict[prefix + 'heading_class_head.bias']
        dir_res_bias = state_dict[prefix + 'heading_residual_head.bias']
        size_cls_bias = state_dict[prefix + 'size_class_head.bias']
        size_res_bias = state_dict[prefix + 'size_residual_head.bias']
        state_dict[prefix + 'conv_reg.bias'] = torch.cat([
            center_bias, dir_cls_bias, dir_res_bias, size_cls_bias,
            size_res_bias
        ])
        del state_dict[prefix + 'center_residual_head.bias']
        del state_dict[prefix + 'heading_class_head.bias']
        del state_dict[prefix + 'heading_residual_head.bias']
        del state_dict[prefix + 'size_class_head.bias']
        del state_dict[prefix + 'size_residual_head.bias']

    model.load_state_dict(state_dict)
    # print(state_dict)

    print(f'{checkpoint_path} loaded in head successfully!!!')

    del checkpoint
    torch.cuda.empty_cache()


def load_checkpoint_net(checkpoint_path, model):
    # Load checkpoint if there is any

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model']

    for k in list(state_dict.keys()):
        # print(k, '\t', state_dict[k].shape)
        # state_dict[k[len("module."):]] = state_dict[k]
        # delete renamed or unused k
        # del state_dict[k]

        if 'backbone_net' in k:
            if 'sa' in k:
                a, b = k.split('.mlp_module')
                if 'bn' in k:
                    key = 'backbone.SA_modules.' + str(
                        (int(a[-1]) - 1)) + '.mlps.0' + b[:7] + b[10:]
                else:
                    key = 'backbone.SA_modules.' + str(
                        (int(a[-1]) - 1)) + '.mlps.0' + b
            else:
                a, b = k.split('.mlp')
                if 'bn' in k:
                    key = 'backbone.FP_modules.' + str(
                        (int(a[-1]) - 1)) + '.mlps' + b[:7] + b[10:]
                else:
                    key = 'backbone.FP_modules.' + str(
                        (int(a[-1]) - 1)) + '.mlps' + b

            state_dict[key] = state_dict[k]
            del state_dict[k]
            continue

        if 'points_obj_cls' in k:
            state_dict['bbox_head.' + k[len('module.'):]] = state_dict[k]

        elif 'proposal_head' in k:
            if '1' in k:
                a = '0.'
                b, c = k[len('module.proposal_head.'):].split('1')
                key = 'bbox_head.conv_pred.shared_convs.layer' + a + b + c
            elif '2' in k:
                a = '1.'
                b, c = k[len('module.proposal_head.'):].split('2')
                key = 'bbox_head.conv_pred.shared_convs.layer' + a + b + c
            else:
                key = 'bbox_head.conv_pred.' + k[len('module.proposal_head.'):]
            state_dict[key] = state_dict[k]

        elif 'decoder_key_proj' in k:
            state_dict['bbox_head.' + k[len('module.'):]] = state_dict[k]

        elif 'decoder_query_proj' in k:
            state_dict['bbox_head.' + k[len('module.'):]] = state_dict[k]

        elif 'decoder_self_posembeds' in k:
            state_dict['bbox_head.' + k[len('module.'):]] = state_dict[k]

        elif 'decoder_cross_posembeds' in k:
            state_dict['bbox_head.' + k[len('module.'):]] = state_dict[k]

        elif 'self_attn' in k:
            a, b = k.split('.self_attn')
            key = 'bbox_head.decoder_layers.' + a[-1] + '.attentions.0.attn' \
                + b
            state_dict[key] = state_dict[k]

        elif 'multihead_attn' in k:
            a, b = k.split('.multihead_attn')
            key = 'bbox_head.decoder_layers.' + a[-1] + '.attentions.1.attn' \
                + b
            state_dict[key] = state_dict[k]

        elif 'linear' in k:
            a, b = k.split('.linear')
            if b[0] == '1':
                c = '0.0'
            else:
                c = '1'
            key = 'bbox_head.decoder_layers.' + a[
                -1] + '.ffns.0.layers.' + c + b[1:]
            state_dict[key] = state_dict[k]

        elif 'norm' in k:
            a, b = k.split('.norm')
            c = str(int(b[0]) - 1)
            key = 'bbox_head.decoder_layers.' + a[-1] + '.norms.' + c + b[1:]
            state_dict[key] = state_dict[k]

        elif 'prediction_heads' in k:
            a = k[len('module.prediction_heads.'):]
            b = a[0:2]
            c = a[2:]
            if '1' in c:
                d, e = c.split('1')
                key = 'bbox_head.prediction_heads.' + b + \
                    'shared_convs.layer0.' + d + e
            elif '2' in c:
                d, e = c.split('2')
                key = 'bbox_head.prediction_heads.' + b + \
                    'shared_convs.layer1.' + d + e
            else:
                key = 'bbox_head.prediction_heads.' + a

            state_dict[key] = state_dict[k]

        del state_dict[k]

    obj_weight = state_dict[
        'bbox_head.conv_pred.objectness_scores_head.weight']
    cls_weight = state_dict['bbox_head.conv_pred.sem_cls_scores_head.weight']
    state_dict['bbox_head.conv_pred.conv_cls.weight'] = torch.cat(
        [obj_weight, cls_weight])
    del state_dict['bbox_head.conv_pred.objectness_scores_head.weight']
    del state_dict['bbox_head.conv_pred.sem_cls_scores_head.weight']

    obj_bias = state_dict['bbox_head.conv_pred.objectness_scores_head.bias']
    cls_bias = state_dict['bbox_head.conv_pred.sem_cls_scores_head.bias']
    state_dict['bbox_head.conv_pred.conv_cls.bias'] = torch.cat(
        [obj_bias, cls_bias])
    del state_dict['bbox_head.conv_pred.objectness_scores_head.bias']
    del state_dict['bbox_head.conv_pred.sem_cls_scores_head.bias']

    center_weight = state_dict[
        'bbox_head.conv_pred.center_residual_head.weight']
    dir_cls_weight = state_dict[
        'bbox_head.conv_pred.heading_class_head.weight']
    dir_res_weight = state_dict[
        'bbox_head.conv_pred.heading_residual_head.weight']
    size_cls_weight = state_dict['bbox_head.conv_pred.size_class_head.weight']
    size_res_weight = state_dict[
        'bbox_head.conv_pred.size_residual_head.weight']

    state_dict['bbox_head.conv_pred.conv_reg.weight'] = torch.cat([
        center_weight, dir_cls_weight, dir_res_weight, size_cls_weight,
        size_res_weight
    ])
    del state_dict['bbox_head.conv_pred.center_residual_head.weight']
    del state_dict['bbox_head.conv_pred.heading_class_head.weight']
    del state_dict['bbox_head.conv_pred.heading_residual_head.weight']
    del state_dict['bbox_head.conv_pred.size_class_head.weight']
    del state_dict['bbox_head.conv_pred.size_residual_head.weight']

    center_bias = state_dict['bbox_head.conv_pred.center_residual_head.bias']
    dir_cls_bias = state_dict['bbox_head.conv_pred.heading_class_head.bias']
    dir_res_bias = state_dict['bbox_head.conv_pred.heading_residual_head.bias']
    size_cls_bias = state_dict['bbox_head.conv_pred.size_class_head.bias']
    size_res_bias = state_dict['bbox_head.conv_pred.size_residual_head.bias']

    state_dict['bbox_head.conv_pred.conv_reg.bias'] = torch.cat([
        center_bias, dir_cls_bias, dir_res_bias, size_cls_bias, size_res_bias
    ])
    del state_dict['bbox_head.conv_pred.center_residual_head.bias']
    del state_dict['bbox_head.conv_pred.heading_class_head.bias']
    del state_dict['bbox_head.conv_pred.heading_residual_head.bias']
    del state_dict['bbox_head.conv_pred.size_class_head.bias']
    del state_dict['bbox_head.conv_pred.size_residual_head.bias']

    for i in range(6):
        prefix = 'bbox_head.prediction_heads.' + str(i) + '.'

        obj_weight = state_dict[prefix + 'objectness_scores_head.weight']
        cls_weight = state_dict[prefix + 'sem_cls_scores_head.weight']
        state_dict[prefix + 'conv_cls.weight'] = torch.cat(
            [obj_weight, cls_weight])
        del state_dict[prefix + 'objectness_scores_head.weight']
        del state_dict[prefix + 'sem_cls_scores_head.weight']

        obj_bias = state_dict[prefix + 'objectness_scores_head.bias']
        cls_bias = state_dict[prefix + 'sem_cls_scores_head.bias']
        state_dict[prefix + 'conv_cls.bias'] = torch.cat([obj_bias, cls_bias])
        del state_dict[prefix + 'objectness_scores_head.bias']
        del state_dict[prefix + 'sem_cls_scores_head.bias']

        center_weight = state_dict[prefix + 'center_residual_head.weight']
        dir_cls_weight = state_dict[prefix + 'heading_class_head.weight']
        dir_res_weight = state_dict[prefix + 'heading_residual_head.weight']
        size_cls_weight = state_dict[prefix + 'size_class_head.weight']
        size_res_weight = state_dict[prefix + 'size_residual_head.weight']
        state_dict[prefix + 'conv_reg.weight'] = torch.cat([
            center_weight, dir_cls_weight, dir_res_weight, size_cls_weight,
            size_res_weight
        ])
        del state_dict[prefix + 'center_residual_head.weight']
        del state_dict[prefix + 'heading_class_head.weight']
        del state_dict[prefix + 'heading_residual_head.weight']
        del state_dict[prefix + 'size_class_head.weight']
        del state_dict[prefix + 'size_residual_head.weight']

        center_bias = state_dict[prefix + 'center_residual_head.bias']
        dir_cls_bias = state_dict[prefix + 'heading_class_head.bias']
        dir_res_bias = state_dict[prefix + 'heading_residual_head.bias']
        size_cls_bias = state_dict[prefix + 'size_class_head.bias']
        size_res_bias = state_dict[prefix + 'size_residual_head.bias']
        state_dict[prefix + 'conv_reg.bias'] = torch.cat([
            center_bias, dir_cls_bias, dir_res_bias, size_cls_bias,
            size_res_bias
        ])
        del state_dict[prefix + 'center_residual_head.bias']
        del state_dict[prefix + 'heading_class_head.bias']
        del state_dict[prefix + 'heading_residual_head.bias']
        del state_dict[prefix + 'size_class_head.bias']
        del state_dict[prefix + 'size_residual_head.bias']

    model.load_state_dict(state_dict)
    # print(state_dict)

    print(f'{checkpoint_path} loaded in net successfully!!!')

    del checkpoint
    torch.cuda.empty_cache()


def test_vote_head():

    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    # _setup_seed(0)
    vote_head_cfg = _get_vote_head_cfg(
        'groupfree3d/groupfree3d_8x8_scannet-3d-18class.py')
    self = build_head(vote_head_cfg)

    load_checkpoint_head('tests/scannet_l6o256.pth', self)

    self.cuda()

    self.eval()

    # x = torch.randn([2, 3])
    # print(x)
    # for param_tensor in self.state_dict():
    #     print(param_tensor)
    #     print(param_tensor,'\t',self.state_dict()[param_tensor].size())
    _setup_seed(0)
    fp_xyz = [torch.rand([2, 256, 3], dtype=torch.float32).cuda()]
    fp_features = [torch.rand([2, 288, 256], dtype=torch.float32).cuda()]
    fp_indices = [torch.randint(0, 128, [2, 256]).cuda()]

    # a = torch.arange(0, 256 * 3 * 0.5, 0.5, dtype=torch.float32)
    # a = a.reshape(256, 3).unsqueeze(0)
    # b = torch.arange(-100, -100 + 256 * 3 * 0.5, 0.5, dtype=torch.float32)
    # b = b.reshape(256, 3).unsqueeze(0)
    # fp_xyz = [torch.cat([a, b], dim=0).cuda()]
    # fp_features = [torch.ones([2, 288, 256], dtype=torch.float32).cuda()]

    # idx_a = torch.arange(0, 256 * 2, 2).unsqueeze(0)
    # idx_b = torch.arange(1, 256 * 2 + 1, 2).unsqueeze(0)
    # fp_indices = [torch.cat([idx_a, idx_b], dim=0).cuda()]

    input_dict = dict(
        fp_xyz=fp_xyz, fp_features=fp_features, fp_indices=fp_indices)

    # test forward
    ret_dict = self(input_dict, 'kps')

    # print(ret_dict['obj_scores_proposal'].shape)
    print(ret_dict['size_res_0'].shape)
    # print(ret_dict['dir_res'].shape)

    # for k, v in ret_dict.items():
    #     # print(k, v.shape)
    #     print(k)

    # print(ret_dict['seeds_obj_cls_logits'])
    # print(ret_dict['seeds_obj_cls_logits'].shape)

    # print(ret_dict['center_residual_proposal'])
    # print(ret_dict['center_residual_proposal'].shape)
    # print(ret_dict['center_proposal'])
    # print(ret_dict['center_proposal'].shape)
    # print(ret_dict['dir_class_proposal'])
    # print(ret_dict['dir_class_proposal'].shape)
    # print(ret_dict['dir_res_norm_proposal'])
    # print(ret_dict['dir_res_norm_proposal'].shape)
    # print(ret_dict['dir_res_proposal'])
    # print(ret_dict['dir_res_proposal'].shape)
    # print(ret_dict['size_class_proposal'])
    # print(ret_dict['size_class_proposal'].shape)
    # print(ret_dict['size_res_norm_proposal'])
    # print(ret_dict['size_res_norm_proposal'].shape)
    # print(ret_dict['size_res_proposal'])
    # print(ret_dict['size_res_proposal'].shape)

    # print(ret_dict['obj_scores_proposal'])
    # print(ret_dict['obj_scores_proposal'].shape)
    # print(ret_dict['sem_scores_proposal'])
    # print(ret_dict['sem_scores_proposal'].shape)

    # print(ret_dict['query_proj'])
    # print(ret_dict['query_proj'].shape)

    # print(ret_dict['query_5'])
    # print(ret_dict['query_5'].shape)

    # print(ret_dict['center_residual_0'])
    # print(ret_dict['center_residual_0'].shape)
    # print(ret_dict['center_5'])
    # print(ret_dict['center_5'].shape)
    # print(ret_dict['dir_class_5'])
    # print(ret_dict['dir_class_5'].shape)
    # print(ret_dict['dir_res_norm_proposal'])
    # print(ret_dict['dir_res_norm_proposal'].shape)
    # print(ret_dict['dir_res_5'])
    # print(ret_dict['dir_res_5'].shape)
    # print(ret_dict['size_class_5'])
    # print(ret_dict['size_class_5'].shape)
    # print(ret_dict['size_res_norm_proposal'])
    # print(ret_dict['size_res_norm_proposal'].shape)
    # print(ret_dict['size_res_5'])
    # print(ret_dict['size_res_5'].shape)

    # print(ret_dict['obj_scores_5'])
    # print(ret_dict['obj_scores_5'].shape)
    # print(ret_dict['sem_scores_5'])
    # print(ret_dict['sem_scores_5'].shape)

    # print(ret_dict['dir_class'])
    # print(ret_dict['dir_res_norm'])
    # print(ret_dict['dir_res'])

    # print(ret_dict['size_class'])

    # assert ret_dict['center'].shape == torch.Size([2, 256, 3])
    # assert ret_dict['obj_scores'].shape == torch.Size([2, 256, 2])
    # assert ret_dict['size_res'].shape == torch.Size([2, 256, 18, 3])
    # assert ret_dict['dir_res'].shape == torch.Size([2, 256, 1])

    # test loss
    points = [torch.rand([40000, 4], device='cuda') for i in range(2)]
    _setup_seed(0)
    gt_bbox1 = torch.rand([10, 7], dtype=torch.float32).cuda()
    gt_bbox2 = torch.rand([10, 7], dtype=torch.float32).cuda()

    gt_bbox1 = LiDARInstance3DBoxes(gt_bbox1)
    gt_bbox2 = LiDARInstance3DBoxes(gt_bbox2)
    gt_bboxes = [gt_bbox1, gt_bbox2]
    # print('gt_bboxes: ', gt_bboxes)

    pts_instance_mask_1 = torch.randint(0, 10, [40000], device='cuda')
    pts_instance_mask_2 = torch.randint(0, 10, [40000], device='cuda')
    pts_instance_mask = [pts_instance_mask_1, pts_instance_mask_2]
    # print(pts_instance_mask)

    pts_semantic_mask_1 = torch.randint(0, 19, [40000], device='cuda')
    pts_semantic_mask_2 = torch.randint(0, 19, [40000], device='cuda')
    pts_semantic_mask = [pts_semantic_mask_1, pts_semantic_mask_2]
    # print(pts_semantic_mask)

    labels_1 = torch.randint(0, 18, [10], device='cuda')
    labels_2 = torch.randint(0, 18, [10], device='cuda')
    gt_labels = [labels_1, labels_2]
    # print(gt_labels)

    losses = self.loss(ret_dict, points, gt_bboxes, gt_labels,
                       pts_semantic_mask, pts_instance_mask)

    assert losses['objectness_loss'] >= 0
    assert losses['semantic_loss'] >= 0
    assert losses['center_loss'] >= 0
    assert losses['dir_class_loss'] >= 0
    assert losses['dir_res_loss'] >= 0
    assert losses['size_class_loss'] >= 0
    assert losses['size_res_loss'] >= 0

    loss_sum = torch.tensor([0.0])
    for k, v in losses.items():
        loss_sum += v

    print('loss_sum: ', loss_sum)


def test_vote_net():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')

    _setup_seed(0)
    vote_net_cfg = _get_detector_cfg(
        'groupfree3d/groupfree3d_8x8_scannet-3d-18class.py')
    self = build_detector(vote_net_cfg).cuda()

    # for param_tensor in self.state_dict():
    #     # print(param_tensor)
    #     print(param_tensor,'\t',self.state_dict()[param_tensor].size())

    # load_checkpoint_net('tests/scannet_l6o256.pth', self)

    self.cuda()

    self.eval()

    _setup_seed(0)

    points_0 = torch.rand([40000, 3], device='cuda')
    points_1 = torch.rand([40000, 3], device='cuda')
    points = [points_0, points_1]

    points_cat = torch.stack(points)

    print(points_cat)

    x = self.extract_feat(points_cat)
    bbox_preds = self.bbox_head(x, 'kps')

    for k, v in bbox_preds.items():
        print(k)


def test_model():
    xyz = np.fromfile('tests/data/sunrgbd/points/000001.bin', np.float32)

    # (B, N, 3)
    xyz = torch.from_numpy(xyz[..., :3]).view(1, -1, 3).cuda()
    # (B, C, N)
    features = xyz.repeat([1, 1, 4]).transpose(1, 2).contiguous().cuda()

    print(features.shape)


if __name__ == '__main__':
    # test_vote_head()
    test_vote_net()
    print('!!!!!!!!!!!!')
