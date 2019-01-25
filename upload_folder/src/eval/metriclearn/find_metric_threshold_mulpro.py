import numpy as np
import os
from scipy import spatial
import gc
import mpi4py.MPI as MPI
import time
import multiprocessing

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

from scipy import spatial


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
    print dist.shape
    return dist

def LCosDis(qry,ref):
    print qry.shape
    print ref.shape
    dist=np.zeros((qry.shape[0],ref.shape[0]))
    for i in xrange(qry.shape[0]):
        for j in xrange(ref.shape[0]):
            dist[i][j]=spatial.distance.cosine(qry[i], ref[j])
            print i,j
    print dist
    return dist


def CalClassificationError_MPI(local_qry_feat, local_qry_label, ref_feat, ref_label, threshold_l, dist=None, fid=None):
    if dist is None:
        dist = LEculidean(local_qry_feat, ref_feat)
    else:
        assert dist.shape == (local_qry_feat.shape[0], ref_feat.shape[0])

    local_qry_num = local_qry_feat.shape[0]
    ref_num = ref_feat.shape[0]
    local_pos_err_num_arr = np.zeros(len(threshold_l))
    local_neg_err_num_arr = np.zeros(len(threshold_l))
    local_pos_num = local_neg_num = 0
    for lqi in range(local_qry_num):
        pos_indices = np.where(ref_label == local_qry_label[lqi])[0]
        neg_indices = np.where(ref_label != local_qry_label[lqi])[0]
        pos_dist = dist[lqi][pos_indices]
        neg_dist = dist[lqi][neg_indices]
        local_pos_num += pos_indices.size
        local_neg_num += neg_indices.size
        for idx, thres in enumerate(threshold_l):
            local_pos_err_num_arr[idx] += (pos_dist >= thres).sum()
            local_neg_err_num_arr[idx] += (neg_dist < thres).sum()
    pos_err_num_arr = comm.allreduce(local_pos_err_num_arr, op=MPI.SUM)
    neg_err_num_arr = comm.allreduce(local_neg_err_num_arr, op=MPI.SUM)
    pos_num = comm.allreduce(local_pos_num, op=MPI.SUM)
    neg_num = comm.allreduce(local_neg_num, op=MPI.SUM)

    pos_err_rate_arr = pos_err_num_arr.astype('float') / pos_num
    neg_err_rate_arr = neg_err_num_arr.astype('float') / neg_num
    log = 'pos pair num {}, neg pair num {}\n'.format(pos_num, neg_num)
    for idx, thres in enumerate(threshold_l):
        log += 'Classification error with threshold {:.4f}: false_pos {:.4f}, false_neg {:.4f}, false_avg {:.4f}\n'.format(
            thres, pos_err_rate_arr[idx], neg_err_rate_arr[idx], 0.5 * (pos_err_rate_arr[idx] + neg_err_rate_arr[idx]))

    if comm_rank == 0:
        print(log)

    if fid is not None:
        fid.write(log + '\n')

    return pos_err_rate_arr, neg_err_rate_arr

global_dist = None
def thread_func(local_qry_num, ref_label, local_qry_label, threshold_intra, threshold_inter,
                beg, interval, save_intra_num, save_inter_num):
    global global_dist
    dist = global_dist
    cur_intra_num = 0
    cur_inter_num = 0
    cur_save_intra_index = range(beg, save_intra_num, interval)
    cur_save_inter_index = range(beg, save_intra_num, interval)
    cur_save_intra_num = len(cur_save_intra_index)
    cur_save_inter_num = len(cur_save_inter_index)
    intra_v = []
    inter_v = []
    fpOutIntra = []
    fpOutInter = []
    intra_num = 0
    inter_num = 0

    for lqi in range(beg, local_qry_num, interval):
        intra_indices = np.where(ref_label == local_qry_label[lqi])[0]
        intra_dist = dist[lqi][intra_indices]
        
        if cur_intra_num < cur_save_intra_num:
            for itor in xrange(len(intra_indices)):
                if intra_dist[itor] > threshold_intra:
                    #fpOutIntra[cur_save_intra_index[cur_intra_num], :] = np.array([lqi, intra_indices[itor]])
                    fpOutIntra.append((lqi, intra_indices[itor]))
                    cur_intra_num += 1
                    if cur_intra_num >= cur_save_intra_num:
                        break
        intra_num += intra_indices.size 
        intra_v.append(intra_dist)
        #intra_v[lqi] = intra_dist

        inter_indices = np.where(ref_label != local_qry_label[lqi])[0]
        inter_dist = dist[lqi][inter_indices]
        
        if cur_inter_num < cur_save_inter_num:
            for itor in xrange(len(inter_indices)):
                if inter_dist[itor] < threshold_inter:# and inter_dist[itor]<3.95:
                    #fpOutInter[cur_save_inter_index[cur_inter_num], :] = np.array([lqi, inter_indices[itor]])
                    fpOutInter.append((lqi, inter_indices[itor]))
                    cur_inter_num += 1
                    if cur_inter_num >= cur_save_inter_num:
                        break                        
        
        inter_num += inter_indices.size
        inter_v.append(inter_dist)
    return intra_v, inter_v, intra_num, inter_num, fpOutIntra, fpOutInter
# added by degang.yang
#from mxnet_face/plugin/metriclearn/python/metriclearn/metric.py
def findMetricThreshold(local_qry_feat, local_qry_label, ref_feat, ref_label, threshold_intra = 5.6, threshold_inter = 4.9, 
            save_inter_num=0, save_intra_num=0, dist=None, fid=None):
    start = time.time() 
    if dist is None:
        dist = LEculidean(local_qry_feat, ref_feat)
    else:
        assert dist.shape == (local_qry_feat.shape[0], ref_feat.shape[0])
    dist[dist <= 0] = 0.0
    global global_dist
    global_dist = dist
    end = time.time()
    print 'Distance Costs %f s' % (end - start)
    start = time.time() 
    #dist = np.sqrt(dist)
    end = time.time()
    print 'Sqrt Costs %f s' % (end - start)

    local_qry_num = local_qry_feat.shape[0]
    ref_num = ref_feat.shape[0]

    intra_num = intra_sum = intra_sum2 = 0
    intra_min = 1E20
    intra_max = 0

    inter_num = inter_sum = inter_sum2 = 0
    inter_min = 1E20
    inter_max = 0

    intra_v = [[]] * local_qry_num
    inter_v = [[]] * local_qry_num
    fpOutIntra = []#np.zeros((save_intra_num,2)) #open('./intra.txt','w')
    fpOutInter = []#np.zeros((save_inter_num,2)) #open('./inter.txt','w')
    start = time.time()
    
    process_num = 12 
    pool_handler = multiprocessing.Pool(processes=process_num)

    results_list = []
    slice_num = 40
    for i in range(slice_num):
        results_list.append( \
            pool_handler.apply_async(thread_func, (local_qry_num, ref_label, local_qry_label, threshold_intra, threshold_inter, \
                i, slice_num, save_intra_num, save_inter_num)))
    pool_handler.close()
    pool_handler.join()

    for result in results_list:
        #import pdb
        #pdb.set_trace()
        result = result.get()
        intra_v.extend(result[0])
        inter_v.extend(result[1])
        intra_num += result[2]
        inter_num += result[3]
        fpOutIntra.extend(result[4])
        fpOutInter.extend(result[5])

    dist = None
    gc.collect()
    end = time.time()
    print 'Statistic Time %f s' % (end - start)
    intra_v = np.hstack(intra_v)
    gc.collect()
    #inter_v = np.sort(np.hstack(inter_v))
    dst_index = []
    FAR = [0.01, 0.001, 0.0001, 0.00001]
    for k in range(len(FAR)):
        dst_index.append(int(FAR[k] * inter_num))
    inter_v = np.hstack(inter_v)
    inter_v = np.partition(inter_v, dst_index)
    # save FAR/GAR pairs
    if fid is not None:
        for num in range(len(inter_v)):
            thr = inter_v[num]
            cnt = len(intra_v[intra_v < thr])
            GAR = float(cnt) / intra_num
            fid.write("%.5f %.5f %.4f\n" % ((num + 1.0) / inter_num, GAR, thr))

    for k in range(len(FAR)):
        num = int(FAR[k] * inter_num)
        thr = inter_v[num]
        cnt = len(intra_v[intra_v < thr])
        GAR = float(cnt) / intra_num
        print("thr:%.4f  FAR:%.5f(%d/%d)  GAR:%.5f(%d/%d)" % (thr, FAR[k], num, inter_num, GAR, cnt, intra_num))
    return fpOutIntra, fpOutInter


def fnEvalval50(sDataFolder):
    fpLabels=open(os.path.join(sDataFolder,'labels.txt'),'r')
    fpFeatures=open(os.path.join(sDataFolder,'feas.txt'),'r')
    labels=[]
    qrys=[]
    while 1:
        linelabel = fpLabels.readline()
        if not linelabel:
            break
        label1=int(linelabel.strip())
        if label1==-1:
            linefea=fpFeatures.readline()
            continue
        labels.append(np.array([label1]))
        linefea=fpFeatures.readline()
        if not (linefea):
            print'mismatch label & feauture?'
            break
        fea1=map(float,linefea.split())
        qrys.append(np.array([fea1]))

    local_qry_label = np.concatenate(labels)
    local_qry_feat = np.concatenate(qrys)
    
    print(local_qry_feat.shape, local_qry_label.shape)
    assert local_qry_feat.shape[0] == local_qry_label.shape[0]
    ref_feat = local_qry_feat#my_Allgatherv(local_qry_feat)
    ref_label = local_qry_feat#my_Allgatherv(local_qry_label)
    print(ref_feat.shape, ref_label.shape)
    assert ref_feat.shape[0] == ref_label.shape[0]
    findMetricThreshold(
        local_qry_feat=local_qry_feat,
        local_qry_label=local_qry_label,
        ref_feat=local_qry_feat,
        ref_label=local_qry_label)

    
if __name__=='__main__':
    fnEvalval50('/data-sdb/yu01.wang/Data/FaceReco/FindBadVals/ReplaceWrongLabel')
    #fnEvalval50('/tmp/xin.wang/ForWY/val_50_feature')




