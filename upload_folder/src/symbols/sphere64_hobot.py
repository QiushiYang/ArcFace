__author__ = 'yu01.wang'
import mxnet as mx

def build_conv_no_bn_blocks(symbol, f, k, s, p, prefix='', init_str='', no_bias=False, se_ratio=0):
    conv = mx.sym.Convolution(symbol, num_filter=f,kernel=(k,k), stride=(s,s), pad=(p,p), name='%s_conv'%prefix + init_str, no_bias=no_bias)
    #act = mx.sym.LeakyReLU(conv, act_type="prelu", name='%s_prelu'%prefix)
    if se_ratio > 0:
        name = prefix
        num_filter = f
        ratio = se_ratio
        squeeze = mx.sym.Pooling(data=conv, global_pool=True, kernel=(7,7), pool_type='avg', name=name+'_squeeze')
        squeeze = mx.sym.Flatten(data=squeeze)
        excitation = mx.symbol.FullyConnected(data=squeeze, num_hidden=int(f*ratio), name=name + '_excitation1')
        excitation = mx.sym.Activation(data=excitation, act_type='relu', name=name + '_excitation1_relu')
        excitation = mx.symbol.FullyConnected(data=excitation, num_hidden=num_filter, name=name + '_excitation2')
        excitation = mx.sym.Activation(data=excitation, act_type='sigmoid', name=name + '_excitation2_sigmoid')
        act = mx.symbol.broadcast_mul(conv, mx.symbol.reshape(data=excitation, shape=(-1, num_filter, 1, 1)))
    act = mx.sym.Activation(conv, act_type='relu')
    return act

def build_bottleneck_conv_blocks(symbol, f, k, s, p, prefix1='', prefix2='', init_str="", no_bias=False, bottle_ratio=0.50, se_ratio=0):
    conv = mx.sym.Convolution(symbol, num_filter=int(f*bottle_ratio), kernel=(1,1), stride=(1,1), pad=(0,0), name='%s_conv_1x1'%prefix1+init_str, no_bias=no_bias)
    act = mx.sym.Activation(conv, act_type='relu')

    conv = mx.sym.Convolution(act, num_filter=int(f*bottle_ratio), kernel=(k,k), stride=(s,s), pad=(p,p), name='%s_conv'%prefix2+init_str, no_bias=no_bias)
    act = mx.sym.Activation(conv, act_type='relu')

    conv = mx.sym.Convolution(act, num_filter=f, kernel=(1,1), stride=(1,1), pad=(0,0), name='%s_conv_1x1'%prefix2+init_str, no_bias=no_bias)
    act = mx.sym.Activation(conv, act_type='relu')

    if se_ratio > 0:
        name = prefix2
        num_filter = f
        ratio = se_ratio
        squeeze = mx.sym.Pooling(data=conv, global_pool=True, kernel=(7,7), pool_type='avg', name=name+'_squeeze')
        squeeze = mx.sym.Flatten(data=squeeze)
        excitation = mx.symbol.FullyConnected(data=squeeze, num_hidden=int(f*ratio), name=name + '_excitation1')
        excitation = mx.sym.Activation(data=excitation, act_type='relu', name=name + '_excitation1_relu')
        excitation = mx.symbol.FullyConnected(data=excitation, num_hidden=num_filter, name=name + '_excitation2')
        excitation = mx.sym.Activation(data=excitation, act_type='sigmoid', name=name + '_excitation2_sigmoid')
        act = mx.symbol.broadcast_mul(conv, mx.symbol.reshape(data=excitation, shape=(-1, num_filter, 1, 1)))

    if s == 1: #dim match
        shortcut = symbol
    else:
        shortcut = mx.sym.Convolution(data=symbol, num_filter=f, kernel=(1,1), stride=(s,s), no_bias=no_bias, name='%s_shortcut'%prefix1)
    return mx.sym.ElementWiseSum(shortcut, act)

def build_two_conv_blocks(symbol, f, k, s, p, prefix1='', prefix2='', init_str="", no_bias=False, se_ratio=0):
    conv = mx.sym.Convolution(symbol, num_filter=f,kernel=(k,k), stride=(s,s), pad=(p,p), name='%s_conv'%prefix1+init_str, no_bias=no_bias)
    #act = mx.sym.LeakyReLU(conv, act_type="prelu", name='%s_prelu'%prefix1)
    act = mx.sym.Activation(conv, act_type='relu')
    conv = mx.sym.Convolution(act, num_filter=f,kernel=(k,k), stride=(s,s), pad=(p,p), name='%s_conv'%prefix2+init_str, no_bias=no_bias)
    #act = mx.sym.LeakyReLU(conv, act_type="prelu", name='%s_prelu'%prefix2)
    if se_ratio > 0:
        print 'Using Se_ratio'
        name = prefix1
        num_filter = f
        ratio = se_ratio
        squeeze = mx.sym.Pooling(data=conv, global_pool=True, kernel=(7,7), pool_type='avg', name=name+'_squeeze')
        squeeze = mx.sym.Flatten(data=squeeze)
        excitation = mx.symbol.FullyConnected(data=squeeze, num_hidden=int(f*ratio), name=name + '_excitation1')
        excitation = mx.sym.Activation(data=excitation, act_type='relu', name=name + '_excitation1_relu')
        excitation = mx.symbol.FullyConnected(data=excitation, num_hidden=num_filter, name=name + '_excitation2')
        excitation = mx.sym.Activation(data=excitation, act_type='sigmoid', name=name + '_excitation2_sigmoid')
        act = mx.symbol.broadcast_mul(conv, mx.symbol.reshape(data=excitation, shape=(-1, num_filter, 1, 1)))
    act = mx.sym.Activation(conv, act_type='relu')
    return act

def resnet_conv5Sphere(model_type, embedding_size, with_bn, fix_fc_layer=False, with_se=False, use_bottleneck=False, dropout_ratio=0.0, channel_type = 0):
    data = mx.sym.Variable("data")
    layer_blocks_20 = {   1 : 1,     2 : 2,      3 : 4,      4 : 1   }
    layer_blocks_36 = {   1 : 2,     2 : 4,      3 : 8,      4 : 2   }
    layer_blocks_64 = {   1 : 3,     2 : 8,      3 : 16,     4 : 3,     5 : 2   }
    layer_blocks_80 = {   1 : 4,     2 : 10,     3 : 20,     4 : 4,     5 : 3   } # 8 + 20 + 40 + 8
    layer_blocks_96 = {   1 : 5,     2 : 12,     3 : 24,     4 : 5,     5 : 4   } # 8 + 20 + 40 + 8

    channel_dict_1 =  {   1 : 64,    2 : 128,    3 : 256,    4 : 512,   5 : 1024 }
    channel_dict_2 =  {   1 : 64,    2 : 256,    3 : 512,    4 : 1024,  5 : 2048 }

    if model_type == 1:
        layer_blocks = layer_blocks_20
    elif model_type == 2:
        layer_blocks = layer_blocks_36
    elif model_type == 3:
        layer_blocks = layer_blocks_64
        se_ratio = 1.0 / 8
    elif model_type == 4:
        layer_blocks = layer_blocks_80
    elif model_type == 5:
        layer_blocks = layer_blocks_96
    else:
        print 'Unknown Model Type in train_info.py'
        assert(0)
    if channel_type == 0:
        channel_dict = channel_dict_1
        use_bottleneck_dict = {1:False, 2:False, 3:True, 4:True, 5:True}
        bottleneck_ratio_dict = {1:0, 2:0, 3:0.5, 4:0.5, 5:0.5}
    else:
        channel_dict = channel_dict_2
        use_bottleneck_dict = {1:False, 2:True, 3:True, 4:True, 5:True}
        bottleneck_ratio_dict = {1:0, 2:0.5, 3:0.25, 4:0.25, 5:0.25}

    se_ratio = 1.0/16
    xa_init = ''#'xavier'
    gau_init = ''#'gaussian'

    channel = channel_dict[1]
    conv1_1 = build_conv_no_bn_blocks(data, f=channel, k=3, s=2, p=1, prefix='conv1_1', init_str=xa_init, se_ratio=se_ratio) #48, 56
    res = conv1_1
    for i in range(layer_blocks[1]):
        conv1_x = build_two_conv_blocks(res, f=channel, k=3, s=1, p=1, prefix1='conv1_%d'%(2*i+2), prefix2='conv1_%d'%(2*i+3), init_str=gau_init, no_bias=False, se_ratio=se_ratio if with_se else 0)
        res = mx.symbol.ElementWiseSum(res, conv1_x)

    for k in range(len(layer_blocks)-1):
        layer_k = k + 2
        channel = channel_dict[layer_k]
        if use_bottleneck and use_bottleneck_dict[layer_k]:
            res = build_bottleneck_conv_blocks(res, f=channel, k=3, s=2, p=1, prefix1='conv%d_%d'%(layer_k, 0), prefix2='conv%d_%d'%(layer_k, 1),  no_bias=False, bottle_ratio=bottleneck_ratio_dict[layer_k], se_ratio=se_ratio)
        else: # 28, 24
            res = build_conv_no_bn_blocks(res, f=channel, k=3, s=2, p=1, prefix='conv%d_1'%layer_k, se_ratio=se_ratio if with_se else 0)

        for i in range(layer_blocks[layer_k]):
            if use_bottleneck and use_bottleneck_dict[layer_k]:
                res = build_bottleneck_conv_blocks(res, f=channel, k=3, s=1, p=1, prefix1='conv%d_%d'%(layer_k, 2*i+2), prefix2='conv%d_%d'%(layer_k, 2*i+3), init_str=gau_init, no_bias=False, bottle_ratio=bottleneck_ratio_dict[layer_k], se_ratio=se_ratio)
            else:
                conv = build_two_conv_blocks(res, f=channel, k=3, s=1, p=1, prefix1='conv%d_%d'%(layer_k, 2*i+2), prefix2='conv%d_%d'%(layer_k, 2*i+3), no_bias=False, se_ratio=se_ratio if with_se else 0)
                res = mx.symbol.ElementWiseSum(res, conv)

    flatten = mx.sym.Flatten(res)
    if dropout_ratio > 0.0:
        flatten = mx.sym.Dropout(flatten, p=dropout_ratio)

    if with_bn:
        flatten = mx.sym.BatchNorm(data=flatten, fix_gamma=False, eps=2e-5, momentum=0.9,
                                sync=True,
                                name='bn1')

    if fix_fc_layer is True:
        mul_lr = "0.0"
        print 'Fixing fc Layer'
    else:
        mul_lr = "1.0"
    if embedding_size == 128:
        fc5 = mx.sym.FullyConnected(flatten, num_hidden=embedding_size, name='fc5_gaussian', no_bias=True,
                attr={"lr_mult":mul_lr})#4
        weight_name = "fc5_gaussian_weight"
    elif embedding_size == 256: # Soly due to we have a such model to finetune from
        fc5 = mx.sym.FullyConnected(flatten, num_hidden=embedding_size, name='fc5_xaiver_%d'%embedding_size, no_bias=True,
                attr={"lr_mult":mul_lr})#4
        weight_name = 'fc5_xaiver_%d_weight'%embedding_size
    else:
        fc5 = mx.sym.FullyConnected(flatten, num_hidden=embedding_size, name='fc5_gaussian_%d'%embedding_size, no_bias=True,
                attr={"lr_mult":mul_lr})#4
        weight_name = 'fc5_gaussian_%d_weight'%embedding_size

    if with_bn:
        fc5 = mx.sym.BatchNorm(data=fc5, fix_gamma=True, eps=2e-5, momentum=0.9,
                                sync=True,
                                name='fc5_bn1')

    return fc5

def get_symbol(args, fix_fc_layer=False):
    return resnet_conv5Sphere(model_type=3, embedding_size=args.emb_size, with_bn = args.with_bn, fix_fc_layer=fix_fc_layer, with_se=True, use_bottleneck=True, dropout_ratio=0, channel_type=1)
