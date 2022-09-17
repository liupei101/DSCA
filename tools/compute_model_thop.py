import torch
import torch.nn as nn
from types import SimpleNamespace
import argparse
from thop import profile, clever_format
from ptflops import get_model_complexity_info

from model import HierNet


def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)
    
    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n
    
    print('# parameters: %d' % num_params)
    print('# trainable parameters: %d' % num_params_train)

def model_setup(args):
    # load model configuration
    print("[info] Got model setting:", args)
    print('[info] {}: initializing model...'.format(args.t))

    # in_dim / hid1_dim / hid2_dim / out_dim
    dims = [1024, 384, 384, 4]
    if 'x5' in args.t:
        scale = 1
        ksize = 5
    else:
        scale = 4
        ksize = 1 if args.m1.lower() == 'fc' else 3 # conv -> ksize=3 / fc -> ksize=1

    if args.t == 'setransurv': # just follow the setting of SeTranSurv
        assert args.m2 == 'identity' and args.m3 == 'Transformer'
    if args.t == 'x5-surv':
        assert args.m2 == 'conv1d'
    if args.m2 == 'conv1d':
        assert args.t == 'x5-surv'
    elif args.m2 == 'sconv':
        assert args.m1 == 'conv'

    if args.t == 'x5-surv' or (args.t == 'x20-surv' and args.m2 != 'capool') or args.t == 'setransurv':
        cfg_emb_backbone = SimpleNamespace(in_dim=dims[0], out_dim=dims[1], scale=scale, dropout=0.6, dw_conv=False, ksize=ksize)
        cfg_tra_backbone = SimpleNamespace(d_model=dims[1], d_out=dims[2], nhead=8, dropout=0.6, num_layers=1)
        model = HierNet.WSIGenericNet(
            dims, args.m2, cfg_emb_backbone, args.m3, cfg_tra_backbone, dropout=0.6, pool='gap'
        )
        model.apply(init_weights) # model parameter init
    elif args.t == 'x20-surv' and args.m2 == 'capool':
        cfg_emb_backbone = SimpleNamespace(in_dim=dims[0], out_dim=dims[1], scale=scale, dropout=0.6, dw_conv=False, ksize=ksize)
        cfg_tra_backbone = SimpleNamespace(d_model=dims[1], d_out=dims[2], nhead=8, dropout=0.6, num_layers=1)
        model = HierNet.WSIGenericCAPNet(
            dims, 'capool', cfg_emb_backbone, args.m3, cfg_tra_backbone, dropout=0.6, pool='gap'
        )
        model.apply(init_weights) # model parameter init
    elif args.t == 'hier-surv':
        cfg_x20_emb = SimpleNamespace(backbone=args.m2, 
            in_dim=dims[0], out_dim=dims[1], scale=4, dropout=0.6, dw_conv=False, ksize=ksize)
        cfg_x5_emb = SimpleNamespace(backbone='conv1d', in_dim=dims[0], out_dim=dims[1], scale=1, dropout=0.6, dw_conv=False, ksize=5)
        cfg_tra_backbone = SimpleNamespace(backbone=args.m3, d_model=dims[1], d_out=dims[2], nhead=8, dropout=0.6, num_layers=1)
        model = HierNet.WSIHierNet(
            dims, cfg_x20_emb, cfg_x5_emb, cfg_tra_backbone, 
            dropout=0.6, pool='gap', join='post', fusion=args.f
        )
    else:
        raise ValueError('check the arguments your passed.')
    
    print('[info] {}: finished model loading'.format(args.t))

    return model

@torch.no_grad()
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()

def input_constructor(args=((1, 200, 1024), 'x20-surv', 'avgpool', 'tuple')):
    bs, N, d = args[0]
    ret_type = args[1]
    pooling  = args[2]
    ret_format = args[3]
    assert ret_format in ['tuple', 'mapping']
    assert ret_type in ['x5-surv', 'x20-surv', 'hier-surv', 'setransurv']
    # generate data for a WSI with 210 patches at 5x magnification
    # The size of test data is same as all baselines to compute GFLOPs fairly.
    feat_x5  = torch.randn(bs, N, d).cuda()
    feat_x20 = torch.randn(bs, 16*N, d).cuda()
    if ret_type == 'setransurv': # sampling strategy in SeTranSurv
        feat_x20 = torch.randn(bs, 600, d).cuda()
        if ret_format == 'tuple':
            return feat_x20,
        else:
            return {'x': feat_x20}
    elif ret_type == 'x5-surv':
        if ret_format == 'tuple':
            return feat_x5,
        else:
            return {'x': feat_x5}
    elif ret_type == 'x20-surv':
        if pooling == 'capool':
            if ret_format == 'tuple':
                return feat_x20, feat_x5
            else:
                return {'x': feat_x20, 'x5': feat_x5}
        else:
            if ret_format == 'tuple':
                return feat_x20,
            else:
                return {'x': feat_x20}
    else:
        if ret_format == 'tuple':
            return feat_x20, feat_x5
        else:
            return {'x20': feat_x20, 'x5': feat_x5}

# IT IS RECOMMENDED TO USE ptflops 
# thop would ignore some parameters and MACs in nn.Transformer
# but ptflops can record them.
parser = argparse.ArgumentParser(description='Configurations for Models.')
parser.add_argument('-a', type=str, choices=['thop', 'ptflops'], default='ptflops')
parser.add_argument('-t', type=str, choices=['x5-surv', 'x20-surv', 'hier-surv', 'setransurv'], default='none')
parser.add_argument('-m1', type=str, choices=['fc', 'conv'], default='fc')
parser.add_argument('-m2', type=str, choices=['conv1d', 'identity', 'avgpool', 'gapool', 'capool', 'sconv'], default='avgpool')
parser.add_argument('-m3', type=str, choices=['Identity', 'Transformer'], default='Transformer')
parser.add_argument('-f', type=str, choices=['cat', 'fusion'], default='cat')

# python3 compute_model_thop.py -t x20-surv -m1 fc -m2 avgpool -m3 Transformer 
if __name__ == '__main__':
    args = parser.parse_args()
    model = model_setup(args)
    print_network(model)
    model = model.cuda()

    N = 210 # Patient '128599' from NLST with 1 slides, 210 patches at 5x, and 3360 patches at 20x

    if args.a == 'thop':
        all_input = input_constructor(args=((1, N, 1024), args.t, args.m2, 'tuple'))
        macs, params = profile(model, inputs=all_input)
    else:
        macs, params = get_model_complexity_info(
            model, ((1, N, 1024), args.t, args.m2, 'mapping'), as_strings=False,
            input_constructor=input_constructor, print_per_layer_stat=True, verbose=True
        )

    print("#Params: {}, #MACs: {}".format(params, macs))
    macs, params = clever_format([macs, params], "%.2f")
    print("#Params: {}, #MACs: {}".format(params, macs))
