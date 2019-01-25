# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

'''
Adapted from https://github.com/tornadomeet/ResNet/blob/master/symbol_resnet.py
Original author Wei Wu
Revised author Mengjia Yan
Implemented the following paper:

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Identity Mappings in Deep Residual Networks"
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import mxnet as mx
import numpy as np
import sklearn
import probe



def Act(data, act_type, name):
    if act_type=='prelu':
      body = mx.sym.LeakyReLU(data = data, act_type='prelu', name = name)
    else:
      body = mx.symbol.Activation(data=data, act_type=act_type, name=name)
    return body

def Conv1(last_conv, act_type):
    use_sync_bn = 1
    mirroring_level = 2
    bn_mom = 0.9
    workspace = 256
    body = last_conv
    body = mx.sym.Convolution(data=body, num_filter=64, kernel=(3,3), stride=(1,1), pad=(1, 1),
                                no_bias=True, name="conv0", workspace=workspace)
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom,
                            sync=True if use_sync_bn else False,
                            attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                            if mirroring_level >= 2 else {},
                            name='bn0')
    body = Act(data=body, act_type=act_type, name='relu0')
    if mirroring_level >= 1:
            body._set_attr(force_mirroring='True')
    return body

def ResidualBlock(last_conv, input_channels, output_channels, act_type, name, stride=1):
    use_sync_bn = 1
    mirroring_level = 2
    bn_mom = 0.9
    workspace = 256
    residual = last_conv
    body = last_conv
    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, 
                           sync=True if use_sync_bn else False,
                           attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                           if mirroring_level >= 2 else {},
                           name=name + '_bn1')
    relu1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')
    if mirroring_level >= 1:
            relu1._set_attr(force_mirroring='True')
    conv1 = mx.sym.Convolution(data=relu1, num_filter=int(output_channels/4), kernel=(1,1), stride=(1,1), pad=(0,0),
                                      no_bias=True, workspace=256, name=name + '_conv1')
    body = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom,
                            sync=True if use_sync_bn else False,
                            attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                            if mirroring_level >= 2 else {}, 
                            name=name + '_bn2')
    body = Act(data=body, act_type=act_type, name=name + '_relu2')
    if mirroring_level >= 1:
            body._set_attr(force_mirroring='True')
    conv2 = mx.sym.Convolution(data=body, num_filter=int(output_channels/4), kernel=(3,3), stride=(stride,stride), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
    body = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, 
                            sync=True if use_sync_bn else False,
                            attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                            if mirroring_level >= 2 else {},
                            name=name + '_bn3')
    body = Act(data=body, act_type=act_type, name=name + '_relu3')
    if mirroring_level >= 1:
            body._set_attr(force_mirroring='True')
    conv3 = mx.sym.Convolution(data=body, num_filter=output_channels, kernel=(1,1), stride=(1,1), pad=(0,0),
                                      no_bias=True, workspace=workspace, name=name + '_conv3')
    if (input_channels != output_channels) or (stride !=1 ):
       residual = mx.sym.Convolution(data=relu1, num_filter=output_channels, kernel=(1,1), stride=(stride,stride), pad=(0,0),
                                      no_bias=True, workspace=workspace, name=name + '_conv4')
    return conv3 + residual

def AttentionModule_stage1(last_conv, input_channels, output_channels, act_type ,name):
    use_sync_bn = 1
    mirroring_level = 2
    bn_mom = 0.9
    workspace = 256
    body = ResidualBlock(last_conv, input_channels, output_channels, act_type, name+'r1')
    out_trunk = ResidualBlock(body, input_channels, output_channels, act_type, name+'r2')
    out_trunk = ResidualBlock(out_trunk, input_channels, output_channels, act_type, name+'r3')
    pool1 = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    mask1 = ResidualBlock(pool1, input_channels, output_channels, act_type, name+'m1')
    skip1 = ResidualBlock(mask1, input_channels, output_channels, act_type, name+'s1')
    pool2 = mx.sym.Pooling(data=mask1, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    mask2 = ResidualBlock(pool2, input_channels, output_channels, act_type, name+'m2')
    skip2 = ResidualBlock(mask2, input_channels, output_channels, act_type, name+'s2')
    pool3 = mx.sym.Pooling(data=mask2, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    mask3 = ResidualBlock(pool3, input_channels, output_channels, act_type, name+'m3_1')
    mask3 = ResidualBlock(mask3, input_channels, output_channels, act_type, name+'m3_2')
    #interp3 = mx.sym.UpSampling(mask3, sample_type='nearest', scale = 2,  workspace=workspace, name=name+'interp3') + mask2
    interp3 = mx.sym.UpSampling(mask3, sample_type='nearest', scale = 2, workspace=workspace, name=name+'interp3') + mask2
    #interp3 = mx.sym.Deconvolution(data=mask3, num_filter=output_channels, kernel=(4,4), stride=(2,2), pad=(1,1), num_group=1,#output_channels,
    #                                  no_bias=True, workspace=256, name=name + 'interp3') + mask2
    out = interp3 + skip2
    mask4 = ResidualBlock(out, input_channels, output_channels, act_type, name+'m4')
    interp2 = mx.sym.UpSampling(mask4, sample_type='nearest', scale = 2,  workspace=workspace, name=name+'interp2') + mask1
    #interp2 = mx.sym.Deconvolution(data=mask4, num_filter=output_channels, kernel=(4,4), stride=(2,2), pad=(1,1), num_group=1,#output_channels,
    #                                  no_bias=True, workspace=256, name=name + 'interp2') + mask1
    out = interp2 + skip1
    mask5 = ResidualBlock(out, input_channels, output_channels, act_type, name+'m5')
    interp1 = mx.sym.UpSampling(mask5, sample_type='nearest', scale = 2, workspace=workspace, name=name+'interp1') + out_trunk
    #interp1 = mx.sym.Deconvolution(data=mask5, num_filter=output_channels, kernel=(4,4), stride=(2,2), pad=(1,1), num_group=1,#output_channels,
    #                                  no_bias=True, workspace=256, name=name + 'interp1') + out_trunk
    mask6 = mx.sym.BatchNorm(data=interp1, fix_gamma=False, eps=2e-5, momentum=bn_mom,
                             sync=True if use_sync_bn else False,
                             attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                             if mirroring_level >= 2 else {},
                             name=name + '_bn6_1')
    mask6 = Act(data=mask6, act_type=act_type, name=name + '_relu6_1')
    if mirroring_level >= 1:
            mask6._set_attr(force_mirroring='True')
    mask6 = mx.sym.Convolution(data=mask6, num_filter=output_channels, kernel=(1,1), stride=(1,1), pad=(0,0),
                                      no_bias=True, workspace=256, name=name + '_conv6_1')
    mask6 = mx.sym.BatchNorm(data=mask6, fix_gamma=False, eps=2e-5, momentum=bn_mom, 
                             sync=True if use_sync_bn else False,
                             attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                             if mirroring_level >= 2 else {},
                             name=name + '_bn6_2')
    mask6 = Act(data=mask6, act_type=act_type, name=name + '_relu6_2')
    if mirroring_level >= 1:
            mask6._set_attr(force_mirroring='True')
    mask6 = mx.sym.Convolution(data=mask6, num_filter=output_channels, kernel=(1,1), stride=(1,1), pad=(0,0),
                                      no_bias=True, workspace=256, name=name + '_conv6_2')
    mask6 = mx.symbol.Activation(data=mask6, act_type='sigmoid', name=name+"_sigmoid6")
    out = out_trunk + mx.symbol.broadcast_mul(out_trunk, mask6)
    out_last = ResidualBlock(out, input_channels, output_channels, act_type, name+'last')
    return out_last

def AttentionModule_stage2(last_conv, input_channels, output_channels, act_type ,name):
    use_sync_bn = 1
    mirroring_level = 2
    bn_mom = 0.9
    workspace = 256
    body = ResidualBlock(last_conv, input_channels, output_channels, act_type, name+'r1')
    out_trunk = ResidualBlock(body, input_channels, output_channels, act_type, name+'r2')
    out_trunk = ResidualBlock(out_trunk, input_channels, output_channels, act_type, name+'r3')
    pool1 = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    mask1 = ResidualBlock(pool1, input_channels, output_channels, act_type, name+'m1')
    skip1 = ResidualBlock(mask1, input_channels, output_channels, act_type, name+'s1')
    pool2 = mx.sym.Pooling(data=mask1, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    mask2 = ResidualBlock(pool2, input_channels, output_channels, act_type, name+'m2_1')
    mask2 = ResidualBlock(mask2, input_channels, output_channels, act_type, name+'m2_2')
    interp2 = mx.sym.UpSampling(mask2, sample_type='nearest', scale = 2, workspace=workspace, name=name+'interp2') + mask1
    #interp2 = mx.sym.Deconvolution(data=mask2, num_filter=output_channels, kernel=(4,4), stride=(2,2), pad=(1,1), num_group=1,#output_channels,
    #                                  no_bias=True, workspace=256, name=name + 'interp2') + mask1
    out = interp2 + skip1
    mask3 = ResidualBlock(out, input_channels, output_channels, act_type, name+'m3')
    interp1 = mx.sym.UpSampling(mask3, sample_type='nearest', scale = 2, workspace=workspace, name=name+'interp1') + out_trunk
    #interp1 = mx.sym.Deconvolution(data=mask3, num_filter=output_channels, kernel=(4,4), stride=(2,2), pad=(1,1), num_group=1,#output_channels,
    #                                  no_bias=True, workspace=256, name=name + 'interp1') + out_trunk
    mask4 = mx.sym.BatchNorm(data=interp1, fix_gamma=False, eps=2e-5, momentum=bn_mom, 
                             sync=True if use_sync_bn else False,
                             attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                             if mirroring_level >= 2 else {},
                             name=name + '_bn6_1')
    mask4 = Act(data=mask4, act_type=act_type, name=name + '_relu6_1')
    if mirroring_level >= 1:
            mask4._set_attr(force_mirroring='True')
    mask4 = mx.sym.Convolution(data=mask4, num_filter=output_channels, kernel=(1,1), stride=(1,1), pad=(0,0),
                                      no_bias=True, workspace=256, name=name + '_conv6_1')
    mask4 = mx.sym.BatchNorm(data=mask4, fix_gamma=False, eps=2e-5, momentum=bn_mom, 
                             sync=True if use_sync_bn else False,
                             attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                             if mirroring_level >= 2 else {},
                             name=name + '_bn6_2')
    mask4 = Act(data=mask4, act_type=act_type, name=name + '_relu6_2')
    if mirroring_level >= 1:
            mask4._set_attr(force_mirroring='True')
    mask4 = mx.sym.Convolution(data=mask4, num_filter=output_channels, kernel=(1,1), stride=(1,1), pad=(0,0),
                                      no_bias=True, workspace=256, name=name + '_conv6_2')
    mask4 = mx.symbol.Activation(data=mask4, act_type='sigmoid', name=name+"_sigmoid6")
    out = out_trunk + mx.symbol.broadcast_mul(out_trunk, mask4)
    out_last = ResidualBlock(out, input_channels, output_channels, act_type, name+'last')
    return out_last

def AttentionModule_stage3(last_conv, input_channels, output_channels, act_type ,name):
    use_sync_bn = 1
    mirroring_level = 2
    bn_mom = 0.9
    workspace = 256
    body = ResidualBlock(last_conv, input_channels, output_channels, act_type, name+'r1')
    out_trunk = ResidualBlock(body, input_channels, output_channels, act_type, name+'r2')
    out_trunk = ResidualBlock(out_trunk, input_channels, output_channels, act_type, name+'r3')
    pool1 = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    mask1 = ResidualBlock(pool1, input_channels, output_channels, act_type, name+'m1_1')
    mask1 = ResidualBlock(mask1, input_channels, output_channels, act_type, name+'s1_2')
    interp1 = mx.sym.UpSampling(mask1, sample_type='nearest', scale = 2, workspace=workspace, name=name+'interp2') + out_trunk
    #interp1 = mx.sym.Deconvolution(data=mask1, num_filter=output_channels, kernel=(4,4), stride=(2,2), pad=(1,1), num_group=1,#output_channels,
    #                                  no_bias=True, workspace=256, name=name + 'interp1') + out_trunk 
    mask4 = mx.sym.BatchNorm(data=interp1, fix_gamma=False, eps=2e-5, momentum=bn_mom, 
                             sync=True if use_sync_bn else False,
                             attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                             if mirroring_level >= 2 else {},
                             name=name + '_bn6_1')
    mask4 = Act(data=mask4, act_type=act_type, name=name + '_relu6_1')
    if mirroring_level >= 1:
            mask4._set_attr(force_mirroring='True')
    mask4 = mx.sym.Convolution(data=mask4, num_filter=output_channels, kernel=(1,1), stride=(1,1), pad=(0,0),
                                      no_bias=True, workspace=256, name=name + '_conv6_1')
    mask4 = mx.sym.BatchNorm(data=mask4, fix_gamma=False, eps=2e-5, momentum=bn_mom, 
                             sync=True if use_sync_bn else False,
                             attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                             if mirroring_level >= 2 else {},
                             name=name + '_bn6_2')
    mask4 = Act(data=mask4, act_type=act_type, name=name + '_relu6_2')
    if mirroring_level >= 1:
            mask4._set_attr(force_mirroring='True')
    mask4 = mx.sym.Convolution(data=mask4, num_filter=output_channels, kernel=(1,1), stride=(1,1), pad=(0,0),
                                      no_bias=True, workspace=256, name=name + '_conv6_2')
    mask4 = mx.symbol.Activation(data=mask4, act_type='sigmoid', name=name+"_sigmoid6")
    out = out_trunk + mx.symbol.broadcast_mul(out_trunk, mask4)
    out_last = ResidualBlock(out, input_channels, output_channels, act_type, name+'last')
    return out_last

def Mpool2(last_conv, name):
    use_sync_bn = 1
    mirroring_level = 2
    bn_mom = 0.9
    body = last_conv
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, 
                            sync=True if use_sync_bn else False,
                            attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                            if mirroring_level >= 2 else {},
                            name=name+'bn')
    body = Act(data=body, act_type='prelu', name=name + 'relu')
    if mirroring_level >= 1:
            body._set_attr(force_mirroring='True')
    pool2 = mx.sym.Pooling(data=body, global_pool=True, kernel=(7, 7), stride=(1,1), pad=(0,0), pool_type='avg')
    return pool2



def resnet_92(num_classes, bottle_neck, **kwargs):
    act_type = kwargs.get('version_act', 'prelu')
    data = mx.sym.Variable(name='data')
    data = mx.sym.identity(data=data, name='id')
    data = data-127.5
    data = data*0.0078125
    #data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    if bottle_neck==1:
        out = Conv1(data,act_type)
        # print(out.data)
        out = ResidualBlock(out, 64, 256, act_type, 'r1')
        out = AttentionModule_stage1(out, 256, 256, act_type, 'a1')
        out = ResidualBlock(out, 256, 512, act_type, 'r2', 2)
        out = AttentionModule_stage2(out, 512, 512, act_type, 'a2_1')
        out = AttentionModule_stage2(out, 512, 512, act_type, 'a2_2')
        out = ResidualBlock(out, 512, 1024, act_type, 'r3', 2)
        # print(out.data)
        out = AttentionModule_stage3(out, 1024, 1024, act_type, 'a3_1')
        out = AttentionModule_stage3(out, 1024, 1024, act_type, 'a3_2')
        out = AttentionModule_stage3(out, 1024, 1024, act_type, 'a3_3')
        out = ResidualBlock(out, 1024, 2048, act_type, 'r4', 2)
        out = ResidualBlock(out, 2048, 2048, act_type, 'r5')
        out = ResidualBlock(out, 2048, 2048, act_type, 'r6')
        out = Mpool2(out,'p1')
        out = mx.sym.FullyConnected(data=out, num_hidden=num_classes, name='fc1')
    else:
        out = Conv1(data,act_type)
        # print(out.data)
        out = ResidualBlock(out, 64, 256, act_type, 'r1')
        out = AttentionModule_stage1(out, 256, 256, act_type, 'a1')
        out = ResidualBlock(out, 256, 512, act_type, 'r2', 2)
        out = AttentionModule_stage2(out, 512, 512, act_type, 'a2_1')
        out = AttentionModule_stage2(out, 512, 512, act_type, 'a2_2')
        out = ResidualBlock(out, 512, 1024, act_type, 'r3', 2)
        # print(out.data)
        out = AttentionModule_stage3(out, 1024, 1024, act_type, 'a3_1')
        out = AttentionModule_stage3(out, 1024, 1024, act_type, 'a3_2')
        out = AttentionModule_stage3(out, 1024, 1024, act_type, 'a3_3')
        out = ResidualBlock(out, 1024, 2048, act_type, 'r4', 2)
        out = ResidualBlock(out, 2048, 2048, act_type, 'r5')
        out = ResidualBlock(out, 2048, 2048, act_type, 'r6')
        out = Mpool2(out,'p1')
        out = mx.sym.FullyConnected(data=out, num_hidden=num_classes, name='fc1')
      
    return out

def resnet_56(num_classes, bottle_neck, **kwargs):
    act_type = kwargs.get('version_act', 'prelu') 
    data = mx.sym.Variable(name='data')
    data = mx.sym.identity(data=data, name='id')
    data = data-127.5
    data = data*0.0078125
    #data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    if bottle_neck==1:
        out = Conv1(data,act_type)
        # print(out.data)
        out = ResidualBlock(out, 64, 256, act_type, 'r1')
        out = AttentionModule_stage1(out, 256, 256, act_type, 'a1')
        out = ResidualBlock(out, 256, 512, act_type, 'r2', 2)
        out = AttentionModule_stage2(out, 512, 512, act_type, 'a2')
        out = ResidualBlock(out, 512, 1024, act_type, 'r3', 2)
        # print(out.data)
        out = AttentionModule_stage3(out, 1024, 1024, act_type, 'a3')
        out = ResidualBlock(out, 1024, 2048, act_type, 'r4', 2)
        out = ResidualBlock(out, 2048, 2048, act_type, 'r5')
        out = ResidualBlock(out, 2048, 2048, act_type, 'r6')
        out = Mpool2(out,'p1')
        out = mx.sym.FullyConnected(data=out, num_hidden=num_classes, name='fc1')
    else:
        out = Conv1(data,act_type)
        # print(out.data)
        out = ResidualBlock(out, 64, 256, act_type, 'r1')
        out = AttentionModule_stage1(out, 256, 256, act_type, 'a1')
        out = ResidualBlock(out, 256, 512, act_type, 'r2', 2)
        out = AttentionModule_stage2(out, 512, 512, act_type, 'a2')
        out = ResidualBlock(out, 512, 1024, act_type, 'r3', 2)
        # print(out.data)
        out = AttentionModule_stage3(out, 1024, 1024, act_type, 'a3')
        out = ResidualBlock(out, 1024, 2048, act_type, 'r4', 2)
        out = ResidualBlock(out, 2048, 2048, act_type, 'r5')
        out = ResidualBlock(out, 2048, 2048, act_type, 'r6')
        out = Mpool2(out,'p1')
        out = mx.sym.FullyConnected(data=out, num_hidden=num_classes, name='fc1')
      
    return out



def get_symbol(num_classes, num_layers, **kwargs):
    """
    Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
    Original author Wei Wu
    Revised author Mengjia Yan
    """
    if num_layers == 92:
      bottle_neck = 1
      return resnet_92(num_classes = num_classes,
                  bottle_neck = bottle_neck,
                  **kwargs)

    elif num_layers == 56:
      bottle_neck = 1
      return resnet_56(num_classes = num_classes,
                  bottle_neck = bottle_neck,
                  **kwargs)

    else:
        raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))

    


