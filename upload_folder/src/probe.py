import mxnet as mx
import numpy as np

class ProbeLayer(mx.operator.CustomOp):
    def __init__(self, ctx, shapes, dtypes, layername):
        self.layername = layername

    def forward(self, is_train, req, in_data, out_data, aux):
        ##print '+++++++++{} size is {}'.format(self.name, data_shape.shape)
        #import pudb;pudb.set_trace()
        #print self.layername, 'forward'
        #print self.layername,'in_data: ', in_data[0][0][0].asnumpy()
        #print '\n'
        #layername= self.layername
        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        print self.layername, 'backward'
        #if np.sum(np.isnan(out_grad[0].asnumpy())) > 0:
        #import pudb;pudb.set_trace()
        print self.layername, list(out_grad[0][0][:100].asnumpy().reshape(-1))
        #layername= self.layername
        print '\n'
        self.assign(in_grad[0], req[0], out_grad[0])


@mx.operator.register("ProbeLayer")
class ProbeLayerProp(mx.operator.CustomOpProp):
    def __init__(self, layername):
        super(ProbeLayerProp, self).__init__(need_top_grad=True)
        self.layername = layername
        self.count = 0

    def list_arguments(self):
        return ['data', ]

    def list_outputs(self):
        return ['output']

    def list_auxiliary_states(self):
        # call them 'bias' for zero initialization
        return []

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        if self.count == 0:
            print '{} size is {}'.format(self.layername, data_shape)
        self.count = self.count + 1
        return [data_shape], [data_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return ProbeLayer(ctx, shapes, dtypes, self.layername)
