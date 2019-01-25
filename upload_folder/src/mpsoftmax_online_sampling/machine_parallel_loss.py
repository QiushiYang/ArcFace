"""
Softmax for multi machines

Added by zhigang.yang
"""
import math
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
import mxnet.autograd as autograd

class SoftmaxModelParallelOperator(object):
    def __init__(self, batch_size, feature_shape, ctx, comm, device_part_size, updater,
                 no_bias=False, margin=True, easy_margin=True, margin_s=1, margin_m=0):
        self.batch_size = batch_size
        self.input_shape = feature_shape
        self.ctx = ctx
        # self.kvstore = kvstore
        self.comm = comm
        self.device_part_size = device_part_size
        self.updater = updater
        self.no_bias = True if margin else no_bias
        self.margin = margin
        self.easy_margin = easy_margin
        self.margin_s = margin_s
        self.margin_m = margin_m
        self.margin_tmp_space = []

        self.num_parts = len(ctx) if isinstance(ctx, list) else 1
        self.part_size = int(device_part_size / self.num_parts)
        self.data_slices = [slice(i*self.part_size, (i+1)*self.part_size) for i in range(self.num_parts)]
        if self.part_size*self.num_parts is not self.batch_size:
            self.data_slices[-1] = slice((self.num_parts-1)*self.part_size, device_part_size)
        self.part_shape = []
        for i in range(self.num_parts):
            self.part_shape.append(self.data_slices[i].stop-self.data_slices[i].start)

        self.local_name = locals()
        # input feature and label
        self.input_feature = nd.empty((self.batch_size,) + feature_shape)
        self.batch_label = nd.empty((self.batch_size,))
        self.local_name['in_feature'] = []
        for i in range(self.num_parts):
            self.local_name['part_in_feature_%s'%i] = nd.empty((self.batch_size,)+self.input_shape, ctx=self.ctx[i])
            self.local_name['in_feature'].append(self.local_name['part_in_feature_%s'%i])
            self.local_name['part_label_%s'%i] = nd.empty((self.batch_size,), ctx=self.ctx[i])

        # fully connected param
        self.local_name['fully_connected_out'] = []
        self.local_name['fully_connected_ingrad'] = []
        self.local_name['weight_param'] = []
        self.local_name['bias_param'] = []
        self.local_name['weight_ingrad'] = []
        self.local_name['bias_ingrad'] = []

        for i in range(self.num_parts):
            self.local_name['part_weight_%s'%i] = nd.empty((self.part_shape[i],)+self.input_shape, ctx=self.ctx[i])
            self.local_name['part_weight2_%s'%i] = nd.empty((self.part_shape[i],)+self.input_shape, ctx=self.ctx[i])
            self.local_name['part_bias_%s'%i] = nd.empty((self.part_shape[i]), ctx=self.ctx[i])

            self.local_name['part_weight_ingrad_%s'%i] = nd.zeros(((self.part_shape[i],)+self.input_shape), ctx=self.ctx[i])
            self.local_name['part_bias_ingrad_%s'%i] = nd.zeros((self.part_shape[i]), ctx=self.ctx[i])

            self.local_name['part_fully_connected_out_%s'%i] = nd.empty((self.batch_size, self.part_shape[i]), ctx=self.ctx[i])
            self.local_name['part_fully_connected_ingrad_%s'%i] = nd.empty((self.batch_size,)+self.input_shape, ctx=self.ctx[i])

            self.local_name['weight_param'].append(self.local_name['part_weight_%s'%i])
            self.local_name['bias_param'].append(self.local_name['part_bias_%s'%i])
            self.local_name['weight_ingrad'].append(self.local_name['part_weight_ingrad_%s'%i])
            self.local_name['bias_ingrad'].append(self.local_name['part_bias_ingrad_%s'%i])
            self.local_name['fully_connected_out'].append(self.local_name['part_fully_connected_out_%s'%i])
            self.local_name['fully_connected_ingrad'].append(self.local_name['part_fully_connected_ingrad_%s'%i])

        # weight L2 param
        if self.margin:
            for i in range(self.num_parts):
                self.local_name['part_weight_tempgrad_%s'%i] = nd.empty(((self.part_shape[i],)+self.input_shape), ctx=self.ctx[i])
                self.local_name['part_weight_sum2_%s'%i] = nd.empty(((self.part_shape[i])), ctx=self.ctx[i])
                self.local_name['part_bias_tempgrad_%s'%i] = nd.empty((self.part_shape[i]), ctx=self.ctx[i])

        # softmax param
        self.total_max = nd.empty((self.batch_size,))
        self.total_sum = nd.empty((self.batch_size,))
        self.local_name['softmax_grad'] = []
        for i in range(self.num_parts):
            self.local_name['part_grad_%s'%i] = nd.empty((self.batch_size, self.part_shape[i]), ctx=self.ctx[i])
            self.local_name['softmax_grad'].append(self.local_name['part_grad_%s'%i])

            self.local_name['part_max_%s'%i] = nd.empty((self.batch_size,), ctx=self.ctx[i])
            self.local_name['part_sum_%s'%i] = nd.empty((self.batch_size,), ctx=self.ctx[i])
            self.local_name['part_total_max_%s'%i] = nd.empty((self.batch_size,), ctx=self.ctx[i])
            self.local_name['part_total_sum_%s'%i] = nd.empty((self.batch_size,), ctx=self.ctx[i])

            self.local_name['part_out_data_%s'%i] = nd.empty((self.batch_size, self.part_shape[i]), ctx=self.ctx[i])
            self.local_name['part_label_%s'%i] = nd.empty((self.batch_size,), ctx=self.ctx[i])
            self.local_name['part_label_index_1_%s'%i] = nd.empty((self.batch_size,), ctx=self.ctx[i])
            self.local_name['part_label_index_2_%s'%i] = nd.empty((self.batch_size,), ctx=self.ctx[i])

        # batch loss
        self.batch_loss = nd.empty((self.batch_size,))
        self.batch_predict = nd.empty((self.batch_size,))

    def set_params(self, fully_param):
        assert 'weight' in fully_param
        assert 'bias' in fully_param
        for i in range(self.num_parts):
            fully_param['weight'][self.data_slices[i],:].copyto(self.local_name['part_weight_%s'%i])
            fully_param['bias'][self.data_slices[i]].copyto(self.local_name['part_bias_%s'%i])

    def load_data(self, input_feature, batch_label):
        for i in range(self.num_parts):
            input_feature.copyto(self.local_name['part_in_feature_%s'%i])
        nd.waitall()

        # feature L2
       #if self.margin:
       #    self.input_feature[:] = input_feature[:]
       #    for i in range(self.num_parts):
       #        self.local_name['part_in_feature_%s'%i][:] = (nd.L2Normalization(self.local_name['part_in_feature_%s'%i])*self.margin_s)[:]
        batch_label.copyto(self.batch_label)

    def margin_forward(self, in_data, margin_s, margin_m):
        gt_index = self._get_gt_index()

        for i in range(self.num_parts):
            if (gt_index[i][0].size) != 0:
                margin_tmp = in_data[i][gt_index[i][0], gt_index[i][1]]
                self.margin_tmp_space.append(margin_tmp.copy())
                tem_data = self._margin_func(in_data[i][gt_index[i][0], gt_index[i][1]], margin_s, margin_m)
                in_data[i][gt_index[i][0], gt_index[i][1]] = tem_data[:]
            else:
                self.margin_tmp_space.append(None)

    def margin_backward(self, in_data, margin_s, margin_m, out_grad):
        gt_index = self._get_gt_index()
        for i in range(self.num_parts):
            if (gt_index[i][0].size) != 0:
                tem_data = self.margin_tmp_space[i]
                #tem_data = in_data[i][gt_index[i][0], gt_index[i][1]].copy()
                tem_data.attach_grad()
                with autograd.record():
                     s = self._margin_func(tem_data, margin_s, margin_m)
                s.backward(out_grad[i][gt_index[i][0], gt_index[i][1]])
                out_grad[i][gt_index[i][0], gt_index[i][1]] = tem_data.grad[:]
        self.margin_tmp_space = []

    def weightl2_forward(self,):
        # fully weight L2
        for i in range(self.num_parts):
            self.local_name['part_weight_%s'%i] = nd.L2Normalization(self.local_name['part_weight_%s'%i], mode='instance')

    def weightl2_backward(self,):
        # grad backward
        for i in range(self.num_parts):
            self.local_name['part_weight_%s'%i].attach_grad()
            with autograd.record():
                s = nd.L2Normalization(self.local_name['part_weight_%s'%i], mode='instance')
            s.backward(self.local_name['part_weight_tempgrad_%s'%i])

            self.local_name['part_weight_ingrad_%s'%i] += self.local_name['part_weight_%s'%i].grad
            #self.local_name['part_weight_ingrad_%s'%i][:] = self.local_name['part_weight_%s'%i].grad[:]

    def fullyconnect_forward(self, in_data):
        if self.no_bias:
            for i in range(self.num_parts):
                nd.FullyConnected(data=in_data[i], weight=self.local_name['part_weight_%s'%i], no_bias=True, num_hidden=self.part_shape[i], out=self.local_name['part_fully_connected_out_%s'%i])
        else:
            for i in range(self.num_parts):
                nd.FullyConnected(data=in_data[i], weight=self.local_name['part_weight_%s'%i], bias=self.local_name['part_bias_%s'%i], num_hidden=self.part_shape[i], no_bias=self.no_bias, out=self.local_name['part_fully_connected_out_%s'%i])

    def fullyconnect_backward(self, in_data, out_grad):
        if self.no_bias:
            for i in range(self.num_parts):
                in_data[i].attach_grad()
                self.local_name['part_weight_%s'%i].attach_grad()
                with autograd.record():
                    nd.FullyConnected(data=in_data[i], weight=self.local_name['part_weight_%s'%i], num_hidden=self.part_shape[i], no_bias=self.no_bias, out=self.local_name['part_fully_connected_out_%s'%i])

                self.local_name['part_fully_connected_out_%s'%i].backward(out_grad[i])
                self.local_name['part_fully_connected_ingrad_%s'%i][:] = in_data[i].grad[:]
                if not self.margin:
                    self.local_name['part_weight_ingrad_%s'%i] += self.local_name['part_weight_%s'%i].grad
                else:
                    self.local_name['part_weight_tempgrad_%s'%i][:] = self.local_name['part_weight_%s'%i].grad[:]
                    #self.local_name['part_weight_ingrad_%s'%i][:] = self.local_name['part_weight_%s'%i].grad[:]
        else:
            for i in range(self.num_parts):
                in_data[i].attach_grad()
                self.local_name['part_weight_%s'%i].attach_grad()
                self.local_name['part_bias_%s'%i].attach_grad()
                with autograd.record():
                    nd.FullyConnected(data=in_data[i], weight=self.local_name['part_weight_%s'%i], bias=self.local_name['part_bias_%s'%i], num_hidden=self.part_shape[i], no_bias=self.no_bias, out=self.local_name['part_fully_connected_out_%s'%i])

                self.local_name['part_fully_connected_out_%s'%i].backward(out_grad[i])
                self.local_name['part_fully_connected_ingrad_%s'%i][:] = in_data[i].grad[:]
                if not self.margin:
                    self.local_name['part_weight_ingrad_%s'%i] += self.local_name['part_weight_%s'%i].grad
                    self.local_name['part_bias_ingrad_%s'%i] += self.local_name['part_bias_%s'%i].grad
                else:
                    self.local_name['part_weight_tempgrad_%s'%i][:] = self.local_name['part_weight_%s'%i].grad[:]
                    self.local_name['part_bias_tempgrad_%s'%i][:] = self.local_name['part_bias_%s'%i].grad[:]

    def fullyconnect_update(self):
        for i in range(self.num_parts):
            self.updater(100000-i, self.local_name['part_weight_ingrad_%s'%i], self.local_name['part_weight_%s'%i])
            self.updater(10000-i, self.local_name['part_bias_ingrad_%s'%i], self.local_name['part_bias_%s'%i])

            self.local_name['part_weight_ingrad_%s'%i][:] = 0.0
            self.local_name['part_bias_ingrad_%s'%i][:] = 0.0


    # softmax forward only to get sum of exp(y-mmax)
    def softmax_forward(self, in_data):
        for i in range(self.num_parts):
            self.local_name['part_max_%s'%i][:] = nd.max(in_data[i], axis=1)[:]
            self.local_name['part_sum_%s'%i][:] = nd.sum(nd.exp(in_data[i]-self.local_name['part_max_%s'%i].reshape((in_data[i].shape[0], 1))), axis=1)[:]
        nd.waitall()
        sum_exp = nd.empty((self.num_parts, self.batch_size))
        max_exp = nd.empty((self.num_parts, self.batch_size))
        for i in range(self.num_parts):
            self.local_name['part_max_%s'%i].copyto(max_exp[i])
            self.local_name['part_sum_%s'%i].copyto(sum_exp[i])
        nd.waitall()
        self.total_max[:] = nd.max(max_exp, axis=0)[:]
        self.total_sum[:] = nd.sum(sum_exp*(nd.exp(max_exp-self.total_max)), axis=0)[:]

    # softmax loss and grad
    def softmax_backward(self, in_data, total_sum, total_max, cpu_label):
        for i in range(self.num_parts):
            total_sum.copyto(self.local_name['part_total_sum_%s'%i])
            total_max.copyto(self.local_name['part_total_max_%s'%i])
        nd.waitall()

        # loss
        for i in range(self.num_parts):
            self.local_name['part_out_data_%s'%i][:] = nd.exp(in_data[i] - self.local_name['part_total_max_%s'%i].reshape((in_data[i].shape[0], 1)))[:]
            self.local_name['part_out_data_%s'%i][:] /= self.local_name['part_total_sum_%s'%i].reshape((in_data[i].shape[0], 1))[:]
        nd.waitall()

        # grad
        gt_index = self._get_gt_index()
        for i in range(self.num_parts):
            self.local_name['part_grad_%s'%i][:] = self.local_name['part_out_data_%s'%i][:]
            if (gt_index[i][0].size) != 0:
                self.local_name['part_grad_%s'%i][gt_index[i][0], gt_index[i][1]] -= 1.0
                self.batch_loss[gt_index[i][0].as_in_context(mx.cpu())] = -nd.log(self.local_name['part_out_data_%s'%i][gt_index[i][0], gt_index[i][1]])[:]

    def forward(self, max_exp, sum_exp):
        if self.margin:
            self.weightl2_forward()
        self.fullyconnect_forward(self.local_name['in_feature'])
        if self.margin:
            self.margin_forward(self.local_name['fully_connected_out'], self.margin_s, self.margin_m)
        self.softmax_forward(self.local_name['fully_connected_out'])

        max_exp[:] = self.total_max[:]
        sum_exp[:] = self.total_sum[:]

    def backward(self, total_max_exp, total_sum_exp):
        self.softmax_backward(self.local_name['fully_connected_out'], total_sum_exp, total_max_exp, self.batch_label)
        if self.margin:
            self.margin_backward(self.local_name['fully_connected_out'], self.margin_s, self.margin_m, self.local_name['softmax_grad'])
        self.fullyconnect_backward(self.local_name['in_feature'], self.local_name['softmax_grad'])
        if self.margin:
            self.weightl2_backward()

    def update_param(self):
        # update param
        self.fullyconnect_update()

    def _get_gt_index(self,):
        gt_index = []
        # rank = self.kvstore.rank if self.kvstore else 0
        rank = self.comm.rank
        for i in range(self.num_parts):
            part_index_start = self.data_slices[i].start + self.device_part_size * rank
            part_index_end = self.data_slices[i].stop + self.device_part_size * rank

            # consider cpu_label==0
            self.local_name['part_label_%s'%i][:] = ((self.batch_label >= part_index_start)*(self.batch_label<part_index_end)*(self.batch_label+1))[:]
            tem_label = self.local_name['part_label_%s'%i].asnumpy()
            part_label_index_1 = nd.array(np.where(tem_label>0), ctx=self.ctx[i])
            part_label_index_2 = nd.array(tem_label[tem_label>0], ctx=self.ctx[i])
            if (part_label_index_1.size) != 0:
                part_label_index_2 -= 1.0
                part_label_index_2 -= part_index_start

            gt_index.append([part_label_index_1, part_label_index_2])
        return gt_index

    def _margin_func(self, in_data, margin_s, margin_m):
        cos_t = in_data/margin_s
        cos_m = math.cos(margin_m)
        sin_m = math.sin(margin_m)
        mm = math.sin(math.pi-margin_m)*margin_m

        #threshold = 0.0
        threshold = math.cos(math.pi-margin_m)
        if self.easy_margin:
            cond = nd.Activation(data=cos_t, act_type='relu')
        else:
            cond_v = cos_t - threshold
            cond = nd.Activation(data=cond_v, act_type='relu')
        body = cos_t*cos_t
        body = 1.0-body
        sin_t = nd.sqrt(body)

        out_data = cos_t*cos_m
        b = sin_t*sin_m
        out_data = out_data - b
        out_data = out_data * margin_s
        if self.easy_margin:
            new_data = in_data
        else:
            new_data = in_data - margin_s * mm
        out_data = nd.where(cond, out_data, new_data)
        return out_data

    def output_copyto(self, in_grad):
        tem_grad = nd.zeros(in_grad.shape)
        sum_grad = nd.zeros(in_grad.shape)
        for i,islice in enumerate(self.data_slices):
            self.local_name['fully_connected_ingrad'][i].copyto(tem_grad)
            sum_grad += tem_grad

        # feature L2 backward
       #if self.margin:
       #    self.input_feature.attach_grad()
       #    with autograd.record():
       #        s = nd.L2Normalization(self.input_feature)*self.margin_s
       #    s.backward(sum_grad)
       #    sum_grad[:] = self.input_feature.grad[:]

        in_grad[:] = sum_grad[:]

    def output_loss(self, batch_loss, islice):
        #print self.batch_loss
        #print islice
        #print batch_loss[islice]
        batch_loss[islice] = self.batch_loss[:]

    def output_predict(self, predict_value, predict_index, islice):
        total_max_index = nd.empty((self.num_parts, self.batch_size))
        total_max_value = nd.empty((self.num_parts, self.batch_size))
        nd_slices = nd.empty((self.num_parts,))
        for i in range(self.num_parts):
            total_max_index[i][:] = nd.argmax(self.local_name['part_out_data_%s'%i], axis=1)[:]
            total_max_value[i][:] = nd.max(self.local_name['part_out_data_%s'%i], axis=1)[:]
            nd_slices[i] = self.data_slices[i].start
        max_value = nd.argmax(total_max_value, axis=0)
        max_index = total_max_index[max_value, nd.arange(self.batch_size)]

        predict_index[islice] = (nd_slices[max_value] + max_index)[:]
        predict_value[islice] = nd.max(total_max_value, axis=0)

    def dumps(self, save_weight, save_bias):
        for i in range(self.num_parts):
            save_weight[self.data_slices[i],:] = self.local_name['part_weight_%s'%i][:]
            save_bias[self.data_slices[i]] = self.local_name['part_bias_%s'%i][:]
