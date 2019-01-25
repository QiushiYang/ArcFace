__author__ = 'mengjia.yan'
import mxnet as mx
import mxnet.ndarray as nd
import time
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))

import numpy as np
import mpi4py.MPI as MPI
import multiprocessing
import gc

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()



def LDOT(feat, proj):
    coord = np.empty((feat.shape[0], proj.shape[1]), dtype=proj.dtype)
    item_size = max(feat.dtype.itemsize, proj.itemsize)
    chunk_sz = int((1 << 30) / item_size / feat.shape[1])
    for si in range(0, feat.shape[0], chunk_sz):
        ei = min(feat.shape[0], si + chunk_sz)
        coord[si:ei] = np.dot(feat[si:ei], proj)
    return coord


def LEculidean(qry, ref):
    qry_sp = (qry ** 2).sum(axis=1).reshape(qry.shape[0], 1)
    ref_sp = (ref ** 2).sum(axis=1).reshape(1, ref.shape[0])
    dist = -2 * LDOT(qry, ref.T)
    dist += qry_sp
    dist += ref_sp
    return dist


def RetrieveTopN(N, qry_feat, ref_feat, L, dist=None):
    # compute dist
    if dist is None:
        dist = LEculidean(qry_feat, ref_feat)
    indices = np.argsort(dist)
    top_indices = indices[:, :N]
    top_dists = []
    for d, i in zip(dist, top_indices):
        top_dists.append(d[i])
    return top_indices, np.vstack(top_dists)

def RetrieveQry(qry_feat, dist=None):
    # compute dist
    if dist is None:
        dist = LEculidean(qry_feat, qry_feat)
    indices = np.argsort(dist)
    top_indices = indices[:, 1:]
    top_dists = []
    for d, i in zip(dist, top_indices):
        top_dists.append(d[i])
    return top_indices, np.vstack(top_dists)

def RetrieveRank1(qry_feat, ref_feat, L, dist=None):
    # compute dist
    if dist is None:
        dist = LEculidean(qry_feat, ref_feat)
    min_LE = np.min(dist,axis = 1)
    #indices = np.argsort(dist)
    #top_indices = indices[:, :N]
    #top_dists = []
    #for d, i in zip(dist, top_indices):
    #    top_dists.append(d[i])
    return min_LE#top_indices, np.vstack(top_dists)



def CalAccuracyRank1_MPI(local_qry_coord, local_qry_label, ref_coord, ref_label, topN_l, gamma_l=None, mAP_NDCG=True,
                        leave_one_out=False, fid=None):
    # from MetricLearningCPP import RetrieveTopN
    labels = []
    hit_num = 0
    total_num = 0
    for label in local_qry_label:
      if label not in labels:
         labels.append(label)
         new_qry = local_qry_coord[local_qry_label == label]
         qry_ref_matrix = RetrieveRank1(new_qry.astype('float'), ref_coord.astype('float'), np.empty(0)) # 1 x qry
         _, qry_qry_matrix = RetrieveQry(new_qry.astype('float')) # qry x (qry-1)
         qry_row = qry_qry_matrix.shape[0]
         qry_col = qry_qry_matrix.shape[1]
         total_newqry = qry_row * qry_col
         total_num = total_num + total_newqry
         qry_ref_matrix = qry_ref_matrix.reshape(1,qry_ref_matrix.shape[0]) # qry x 1
         qry_ref_matrix = np.broadcast_to(qry_ref_matrix.T,(qry_row,qry_col)) # qry x (qry - 1)
         hit_newqry = np.sum(qry_qry_matrix <= qry_ref_matrix)
         hit_num = hit_num + hit_newqry
    LQN = local_qry_label.size
    QN = comm.allreduce(LQN, op=MPI.SUM)
    log = 'TopK accuracy and hit rate: qry num {}, ref num {}, ref cate num {}, total qry num {}, rank1 is {:.4f}, hit num is {}\n'.format(QN, ref_label.size, np.unique(ref_label).size, total_num, float(hit_num)/total_num, hit_num)
    print log


class NdcgEval_rank(mx.metric.EvalMetric):
    """Calculate NDCG metrics, for metric learning"""

    def __init__(self, topk):
        super(NdcgEval_rank, self).__init__('ndcg')
        self.topk = topk

    def update_1(self, labels, preds):
        #print('update qrys')
        assert len(labels) == 1
        assert len(preds) == 1
        pred = preds[0].asnumpy()
        label = labels[0].asnumpy().astype('int32')
        self.qrys.append(pred)
        self.qrys_labels.append(label)

    def update_2(self, labels, preds):
        #print('update refs')
        assert len(labels) == 1
        assert len(preds) == 1
        pred = preds[0].asnumpy()
        label = labels[0].asnumpy().astype('int32')
        self.refs.append(pred)
        self.refs_labels.append(label)

    def reset(self):
        self.qrys_labels = []
        self.qrys = []
        self.refs_labels = []
        self.refs = []

    def get(self):
        #import pudb;pudb.set_trace()
        local_qry_feat = np.concatenate(self.qrys)
        qrys_norm = np.linalg.norm(local_qry_feat)
        local_qry_feat = local_qry_feat / qrys_norm
        local_qry_label = np.concatenate(self.qrys_labels)
        #print(local_qry_feat.shape, local_qry_label.shape)
        assert local_qry_feat.shape[0] == local_qry_label.shape[0]
        local_ref_feat = np.concatenate(self.refs)
        refs_norm = np.linalg.norm(local_ref_feat)
        local_ref_label = np.concatenate(self.refs_labels)
        ref_feat = local_ref_feat / refs_norm  #my_Allgatherv(local_qry_feat)#local_qry_feat#my_Allgatherv(local_qry_feat)
        ref_label = local_ref_label  #my_Allgatherv(local_qry_label)#local_qry_feat#my_Allgatherv(local_qry_label)
        print(ref_feat.shape, ref_label.shape)
        assert ref_feat.shape[0] == ref_label.shape[0]
        if ref_feat.shape[0] == 12000 or ref_feat.shape[0] == 11904 or \
            ref_feat.shape[0]/2 == 12000 or ref_feat.shape[0]/2 == 11904:
            return (self.name + "_lfw", 0.0)
        '''
        st = time.time()
        CalKNNError_MPI(
              local_qry_feat=local_qry_feat,
              local_qry_label=local_qry_label,
              ref_feat=ref_feat,
              ref_label=ref_label,
              K=self.topk,
              leave_one_out=True)
          print('time spent {} sec'.format(time.time() - st))
       '''
        st = time.time()
        #if ref_feat.shape[0] == 19913:
        CalAccuracyRank1_MPI(
            local_qry_coord=local_qry_feat,
            local_qry_label=local_qry_label,
            ref_coord=ref_feat,
            ref_label=ref_label,
            topN_l=[1, 2, 5, 10, 100],
            leave_one_out=True)
        print('time spent {} sec'.format(time.time() - st))
        #return (self.name, mean_nDCG)
        return (self.name, 1.0)



def rank_data(probe_data,  gallery_data,  part_size,
               executor_manager,  epoch):
    eval_metric = NdcgEval_rank(10) 
    #import pudb;pudb.set_trace()
    eval_metric.reset()
    probe_data.reset()
    #gallerya_data.reset()
    gallery_data.reset()
    print('probe data intial...')
    for i, eval_batch in enumerate(probe_data):
        assert probe_data.batch_size == part_size
        #out_shape = tuple([probe_data.batch_size] +
                          #list(executor_manager.output_shapes[0][1][1:]))
        #cpu_out_feature = nd.zeros(out_shape)
        #executor_manager.load_data_batch(eval_batch)
        executor_manager.forward(eval_batch, is_train=False)
        cpu_out_feature = executor_manager.get_outputs()
        #executor_manager.output_copyto(cpu_out_feature)
        #executor_manager.update_metric(eval_metric, eval_batch)

        if eval_batch.pad > 0:
            real_batch_size = probe_data.batch_size - eval_batch.pad
            eval_metric.update_1(
                [a[:real_batch_size] for a in eval_batch.label],
                [a[:real_batch_size] for a in [cpu_out_feature]])
        else:
            eval_metric.update_1(eval_batch.label, [cpu_out_feature[0]])

    print('gallery data intial...')
    for i, eval_batch in enumerate(gallery_data):
        #print('i:', i)
        assert gallery_data.batch_size == part_size
        #out_shape = tuple([gallery_data.batch_size] +
                          #list(executor_manager.output_shapes[0][1][1:]))
        #cpu_out_feature = nd.zeros(out_shape)
        #executor_manager.load_data_batch(eval_batch)
        executor_manager.forward(eval_batch, is_train=False)
        #executor_manager.output_copyto(cpu_out_feature)
        cpu_out_feature = executor_manager.get_outputs()

        if eval_batch.pad > 0:
            real_batch_size = gallery_data.batch_size - eval_batch.pad
            eval_metric.update_2(
                [a[:real_batch_size] for a in eval_batch.label],
                [a[:real_batch_size] for a in [cpu_out_feature]])
        else:
            eval_metric.update_2(eval_batch.label, [cpu_out_feature[0]])

    name_value = eval_metric.get_name_value()
    #for name, value in name_value:
    #    print('Epoch[%d] eval data-facescrub Validation-%s=%f', epoch,
    #                name, value)

