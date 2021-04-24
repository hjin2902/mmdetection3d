import copy
import numpy as np
import pytest
import random
import torch
from os.path import dirname, exists, join

# from mmdet3d.core.bbox import (Box3DMode, DepthInstance3DBoxes,
#                                LiDARInstance3DBoxes)
from mmdet3d.models.builder import build_head

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


def test_vote_head():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    _setup_seed(0)
    vote_head_cfg = _get_vote_head_cfg(
        'groupfree3d/groupfree3d_8x8_scannet-3d-18class.py')
    self = build_head(vote_head_cfg).cuda()

    for param_tensor in self.state_dict():
        print(param_tensor)
        # print(param_tensor,'\t',self.state_dict()[param_tensor].size())

    # fp_xyz = [torch.rand([2, 256, 3], dtype=torch.float32).cuda()]
    # fp_features = [torch.rand([2, 288, 256], dtype=torch.float32).cuda()]
    # fp_indices = [torch.randint(0, 128, [2, 256]).cuda()]

    a = torch.arange(0, 256 * 3 * 0.5, 0.5, dtype=torch.float32)
    a = a.reshape(256, 3).unsqueeze(0)
    b = torch.arange(-100, -100 + 256 * 3 * 0.5, 0.5, dtype=torch.float32)
    b = b.reshape(256, 3).unsqueeze(0)
    fp_xyz = [torch.cat([a, b], dim=0).cuda()]
    fp_features = [torch.ones([2, 288, 256], dtype=torch.float32).cuda()]

    idx_a = torch.arange(0, 256 * 2, 2).unsqueeze(0)
    idx_b = torch.arange(1, 256 * 2 + 1, 2).unsqueeze(0)
    fp_indices = [torch.cat([idx_a, idx_b], dim=0).cuda()]

    input_dict = dict(
        fp_xyz=fp_xyz, fp_features=fp_features, fp_indices=fp_indices)

    # test forward
    ret_dict = self(input_dict, 'kps')
    # print(ret_dict['center'])

    print(ret_dict['center'].shape)
    print(ret_dict['obj_scores'].shape)
    print(ret_dict['size_res'].shape)
    print(ret_dict['dir_res'].shape)

    # for k, v in ret_dict.items():
    #     print(k, v.shape)

    # print(ret_dict['center'])

    # print(ret_dict['dir_class'])
    # print(ret_dict['dir_res_norm'])
    # print(ret_dict['dir_res'])

    # print(ret_dict['size_class'])

    # assert ret_dict['center'].shape == torch.Size([2, 256, 3])
    # assert ret_dict['obj_scores'].shape == torch.Size([2, 256, 2])
    # assert ret_dict['size_res'].shape == torch.Size([2, 256, 18, 3])
    # assert ret_dict['dir_res'].shape == torch.Size([2, 256, 1])


if __name__ == '__main__':
    test_vote_head()
