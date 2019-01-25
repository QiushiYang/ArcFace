import mxnet as mx
import mxnet.ndarray as nd
import math

def test_func(weight, label, embedding, num_classes, easy_margin, s, m):
    gt_label = label
    _weight = nd.L2Normalization(weight, mode='instance')
    #nembedding = nd.L2Normalization(embedding, mode='instance', name='fc1n')*s
    fc7 = nd.FullyConnected(data=embedding, weight = _weight, no_bias = True,
                               num_hidden=num_classes, name='fc7')
    zy = nd.pick(fc7, gt_label, axis=1)
    cos_t = zy/s
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = math.sin(math.pi-m)*m
    #threshold = 0.0
    threshold = math.cos(math.pi-m)
    if easy_margin:
        cond = nd.Activation(data=cos_t, act_type='relu')
    else:
        cond_v = cos_t - threshold
        cond = nd.Activation(data=cond_v, act_type='relu')
    body = cos_t*cos_t
    body = 1.0-body
    sin_t = nd.sqrt(body)
    new_zy = cos_t*cos_m
    b = sin_t*sin_m
    new_zy = new_zy - b
    new_zy = new_zy*s
    if easy_margin:
        zy_keep = zy
    else:
        zy_keep = zy - s*mm
    new_zy = nd.where(cond, new_zy, zy_keep)

    diff = new_zy - zy
    diff = nd.expand_dims(diff, 1)
    gt_one_hot = nd.one_hot(gt_label, depth=num_classes, on_value = 1.0, off_value = 0.0)
    body = nd.broadcast_mul(gt_one_hot, diff)
    fc7 = fc7+body
    out = nd.SoftmaxOutput(data=fc7, label = gt_label, name='softmax',
            normalization='valid')
    return out
