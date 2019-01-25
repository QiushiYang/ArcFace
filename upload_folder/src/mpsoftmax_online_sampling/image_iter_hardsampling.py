from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import logging
import sys
import numbers
import math
import time
#import sklearn
import datetime
import numpy as np
import cv2
import time

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import io
from mxnet import recordio
from mxnet import context
import mxnet.gluon.data.dataloader as dataloader
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
#import face_preprocess
import multiprocessing
import threading
import Queue

logger = logging.getLogger()


class FaceImageIter(io.DataIter):

    def __init__(self, batch_size, data_shape,
                 path_imgrec = None,
                 shuffle=False, aug_list=None, mean = None,
                 rand_mirror = False, cutoff = 0,
                 data_name='data', label_name='softmax_label', if_sample=True,
                 update_buffer=8, max_weight_gain=1.2, min_weight_gain=0.8,
                 max_weight_ave=100, min_weight_ave=0.01,
                 hard_ratio=0.5, **kwargs):
        super(FaceImageIter, self).__init__()
        assert path_imgrec
        if path_imgrec:
            logging.info('loading recordio %s...',
                         path_imgrec)
            path_imgidx = path_imgrec[0:-4]+".idx"
            self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
            s = self.imgrec.read_idx(0)
            header, _ = recordio.unpack(s)
            if header.flag>0:
              print('header0 label', header.label)
              self.header0 = (int(header.label[0]), int(header.label[1]))
              #assert(header.flag==1)
              self.imgidx = range(1, int(header.label[0]))
              self.id2range = {}
              self.seq_identity = range(int(header.label[0]), int(header.label[1]))
              for identity in self.seq_identity:
                s = self.imgrec.read_idx(identity)
                header, _ = recordio.unpack(s)
                a,b = int(header.label[0]), int(header.label[1])
                self.id2range[identity] = (a,b)
                count = b-a
              print('id2range', len(self.id2range))
            else:
              self.imgidx = list(self.imgrec.keys)
            if shuffle:
              self.seq = self.imgidx
              self.oseq = self.imgidx
              print(len(self.seq))
            else:
              self.seq = None

        self.mean = mean
        self.nd_mean = None
        if self.mean:
          self.mean = np.array(self.mean, dtype=np.float32).reshape(1,1,3)
          self.nd_mean = mx.nd.array(self.mean).reshape((1,1,3))

        self.check_data_shape(data_shape)
        self.provide_data = [(data_name, (batch_size,) + data_shape)]
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.shuffle = shuffle
        self.image_size = '%d,%d'%(data_shape[1],data_shape[2])
        self.rand_mirror = rand_mirror
        print('rand_mirror', rand_mirror)
        self.cutoff = cutoff
        self.provide_label = [(label_name, (batch_size,))]
        #print(self.provide_label[0][1])
        self.cur = 0
        self.reset()

        # for batch update
        self.if_sample = if_sample
        self.prob_step = (max_weight_gain - min_weight_gain) / batch_size
        self.np_update_prob = np.arange(min_weight_gain, max_weight_gain, self.prob_step)
        self.update_loss_queue = Queue.Queue(maxsize = update_buffer)
        self.update_index_queue = Queue.Queue(maxsize = update_buffer)
        self.weighted_prob = np.ones((len(self.imgidx)))/len(self.imgidx)
        self.max_weight_ave = max_weight_ave
        self.min_weight_ave = min_weight_ave
        self.hard_ratio = hard_ratio
        self.hard_num = int(self.batch_size * self.hard_ratio)
        self.random_num = int(self.batch_size - self.hard_num)

        if self.if_sample:
            print("Using hard sampling")
        else:
            print("Using random sampling")

    @property
    def provide_loss_queue(self):
        return self.update_loss_queue

    @property
    def provide_index_queue(self):
        return self.update_index_queue

    def update_prob(self, update_index, update_loss):
        np_update_index = update_index.asnumpy()
        np_update_loss = update_loss.asnumpy()
        arg_loss = np.argsort(np_update_loss, axis=0)
        sort_index_by_loss = np_update_index[arg_loss]
        sort_index_by_loss = sort_index_by_loss.astype(int)-1
        self.weighted_prob[sort_index_by_loss] *= self.np_update_prob

        weight_ave = np.mean(self.weighted_prob)
        max_weight_prob = weight_ave * self.max_weight_ave
        min_weight_prob = weight_ave * self.min_weight_ave
        self.weighted_prob = np.maximum(self.weighted_prob, self.min_weight_ave)
        self.weighted_prob = np.minimum(self.weighted_prob, self.max_weight_ave)

        self.weighted_prob /= np.sum(self.weighted_prob)

    def sample_seq(self):
        sample_index = np.arange(len(self.seq))
        # hard sampling
        hard_sample = np.random.choice(sample_index, self.hard_num, replace=False, p=self.weighted_prob)

        # random sampling
        random_prob = np.ones(len(self.seq)) / (len(self.seq)-len(hard_sample))
        random_prob[hard_sample] = 0.0
        random_sample = np.random.choice(sample_index, self.random_num, replace=False, p=random_prob)

        final_sample = np.append(hard_sample, random_sample)
        np.random.shuffle(final_sample)
        return final_sample

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        print('call reset()')
        self.cur = 0
        if self.seq is None and self.imgrec is not None:
            self.imgrec.reset()

    @property
    def num_samples(self):
        return len(self.seq)

    def next_sample(self, index=None, lock=None, imgrec=None):
        """Helper function for reading in next sample."""
        #set total batch size, for example, 1800, and maximum size for each people, for example 45
        if self.seq is not None:
          while True:
            idx = self.seq[index]
            if imgrec is not None:
              lock.acquire()
              s = imgrec.read_idx(idx)
              lock.release()
              header, img = recordio.unpack(s)
              label = header.label
              if not isinstance(label, numbers.Number):
                label = label[0]
              return label, img, None, None, header.id
            elif self.imgrec is not None:
              if self.cur >= len(self.seq):
                  raise StopIteration
              idx = self.seq[self.cur]
              self.cur += 1

              s = self.imgrec.read_idx(idx)
              header, img = recordio.unpack(s)
              label = header.label
              if not isinstance(label, numbers.Number):
                label = label[0]
              return label, img, None, None, header.id
            else:
              label, fname, bbox, landmark = self.imglist[idx]
              return label, self.read_image(fname), bbox, landmark, None
        else:
            s = self.imgrec.read()
            if s is None:
                raise StopIteration
            header, img = recordio.unpack(s)
            return header.label, img, None, None, header.id

    def brightness_aug(self, src, x):
      alpha = 1.0 + random.uniform(-x, x)
      src *= alpha
      return src

    def contrast_aug(self, src, x):
      alpha = 1.0 + random.uniform(-x, x)
      coef = np.array([[[0.299, 0.587, 0.114]]])
      gray = src * coef
      gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
      src *= alpha
      src += gray
      return src

    def saturation_aug(self, src, x):
      alpha = 1.0 + random.uniform(-x, x)
      coef = np.array([[[0.299, 0.587, 0.114]]])
      gray = src * coef
      gray = np.sum(gray, axis=2, keepdims=True)
      gray *= (1.0 - alpha)
      src *= alpha
      src += gray
      return src

    def color_aug(self, img, x):
      augs = [self.brightness_aug, self.contrast_aug, self.saturation_aug]
      random.shuffle(augs)
      for aug in augs:
        #print(img.shape)
        img = aug(img, x)
        #print(img.shape)
      return img

    def mirror_aug(self, img):
      _rd = random.randint(0,1)
      if _rd==1:
        for c in xrange(img.shape[2]):
          img[:,:,c] = np.fliplr(img[:,:,c])
      return img

    def next(self, lock=None, imgrec=None, final_seq=None):
        """Returns the next batch of data."""
        # batch callback
        if final_seq is None:
            if self.update_loss_queue.empty():
                final_seq = np.random.choice(np.arange(len(self.seq)), self.batch_size, replace=False)
            else:
                update_loss = self.update_loss_queue.get()
                update_index = self.update_index_queue.get()
                if not self.if_sample:
                    final_seq = np.random.choice(np.arange(len(self.seq)), self.batch_size, replace=False)
                else:
                    self.update_prob(update_index, update_loss)
                    final_seq = self.sample_seq()

        batch_size = self.batch_size
        c, h, w = self.data_shape
        batch_data = nd.empty((batch_size, c, h, w))
        batch_index = nd.empty((batch_size,))
        if self.provide_label is not None:
            batch_label = nd.empty(self.provide_label[0][1])
        i = 0
        try:
            while i < batch_size:
                label, s, bbox, landmark, only_idx= self.next_sample(final_seq[i], lock, imgrec)
                _data = self.imdecode(s)
                if self.rand_mirror:
                  _rd = random.randint(0,1)
                  if _rd==1:
                    _data = mx.ndarray.flip(data=_data, axis=1)
                if self.nd_mean is not None:
                    _data = _data.astype('float32')
                    _data -= self.nd_mean
                    _data *= 0.0078125
                if self.cutoff>0:
                  centerh = random.randint(0, _data.shape[0]-1)
                  centerw = random.randint(0, _data.shape[1]-1)
                  half = self.cutoff//2
                  starth = max(0, centerh-half)
                  endh = min(_data.shape[0], centerh+half)
                  startw = max(0, centerw-half)
                  endw = min(_data.shape[1], centerw+half)
                  _data = _data.astype('float32')
                  #print(starth, endh, startw, endw, _data.shape)
                  _data[starth:endh, startw:endw, :] = 127.5
                data = [_data]
                try:
                    self.check_valid_image(data)
                except RuntimeError as e:
                    logging.debug('Invalid image, skipping:  %s', str(e))
                    continue
                #data = self.augmentation_transform(data)
                for datum in data:
                    assert i < batch_size, 'Batch size must be multiples of augmenter output length'
                    #print(datum.shape)
                    batch_data[i][:] = self.postprocess_data(datum)
                    batch_label[i][:] = label
                    batch_index[i][:] = only_idx
                    i += 1
        except StopIteration:
            if i<batch_size:
                raise StopIteration
        return io.DataBatch([batch_data], [batch_label], batch_size-i, [batch_index])

    def check_data_shape(self, data_shape):
        """Checks if the input data shape is valid"""
        if not len(data_shape) == 3:
            raise ValueError('data_shape should have length 3, with dimensions CxHxW')
        if not data_shape[0] == 3:
            raise ValueError('This iterator expects inputs to have 3 channels.')

    def check_valid_image(self, data):
        """Checks if the input data is valid"""
        if len(data[0].shape) == 0:
            raise RuntimeError('Data shape is wrong')

    def imdecode(self, s):
        """Decodes a string or byte string to an NDArray.
        See mx.img.imdecode for more details."""
        img = mx.image.imdecode(s) #mx.ndarray
        return img

    def read_image(self, fname):
        """Reads an input image `fname` and returns the decoded raw bytes.

        Example usage:
        ----------
        >>> dataIter.read_image('Face.jpg') # returns decoded raw bytes.
        """
        with open(os.path.join(self.path_root, fname), 'rb') as fin:
            img = fin.read()
        return img

    def augmentation_transform(self, data):
        """Transforms input data with specified augmentation."""
        for aug in self.auglist:
            data = [ret for src in data for ret in aug(src)]
        return data

    def postprocess_data(self, datum):
        """Final postprocessing step before image is loaded into the batch."""
        return nd.transpose(datum, axes=(2, 0, 1))

class FaceImageIterList(io.DataIter):
  def __init__(self, iter_list):
    assert len(iter_list)>0
    self.provide_data = iter_list[0].provide_data
    self.provide_label = iter_list[0].provide_label
    self.iter_list = iter_list
    self.cur_iter = None

  def reset(self):
    self.cur_iter.reset()

  def next(self):
    self.cur_iter = random.choice(self.iter_list)
    while True:
      try:
        ret = self.cur_iter.next()
      except StopIteration:
        self.cur_iter.reset()
        continue
      return ret

class PrefetchProcessIter(io.DataIter):
    def __init__(self, data_iter, prefetch_process=4, capacity=4):
        super(PrefetchProcessIter, self).__init__()
        assert data_iter is not None
        self.data_iter = data_iter
        self.batch_size = self.provide_data[0][1][0]
        self.epoch_size = self.data_iter.num_samples/self.batch_size
        self.next_iter = 0
        self.prefetch_process = prefetch_process

        self._data_queue = dataloader.Queue(maxsize=capacity)
        self._data_buffer = Queue.Queue(maxsize=capacity*2)

        self.prefetch_reset_event = multiprocessing.Event()
        self.next_reset_event = threading.Event()

        self.lock = multiprocessing.Lock()
        self.imgrec = self.data_iter.imgrec

        def prefetch_func(data_queue, event):
            np.random.seed(random.randint(0, 2**32))
            while True:
                if event.is_set():
                    final_seq = self.data_iter.sample_seq()
                    next_data = self.data_iter.next(self.lock, self.imgrec, final_seq)
                    data_queue.put((dataloader.default_mp_batchify_fn(next_data.data[0]),
                                    dataloader.default_mp_batchify_fn(next_data.label[0]),
                                    dataloader.default_mp_batchify_fn(next_data.index[0])))

        def next_func(data_queue, event):
            while True:
                if event.is_set():
                    batch, label, index = data_queue.get(block=True)
                    batch = dataloader._as_in_context(batch, context.cpu())
                    label = dataloader._as_in_context(label, context.cpu())
                    index = dataloader._as_in_context(index, context.cpu())
                    label = label.reshape((label.shape[0],))
                    index = index.reshape((index.shape[0],))
                    self._data_buffer.put((batch, label, index))

        # producer next
        self.produce_lst = []
        for ith in range(prefetch_process):
            p_process = multiprocessing.Process(target=prefetch_func,
                                                args=(self._data_queue, self.prefetch_reset_event))
            p_process.daemon = True
            p_process.start()
            self.produce_lst.append(p_process)

        # consumer get
        self.data_buffer = {}
        self.prefetch_thread = threading.Thread(target=next_func,
                                                args=(self._data_queue, self.next_reset_event))
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

        # first epoch
        self.data_iter.reset()
        self.prefetch_reset_event.set()
        self.next_reset_event.set()

    def __del__(self):
        self.__clear_queue()

        for i_process in self.produce_lst:
            i_process.join()
        self.prefetch_thread.join()

    def __clear_queue(self):
        """ clear the queue"""
        while True:
            try:
                self._data_queue.get_nowait()
            except:
                break
        while True:
            try:
                self._data_buffer.get_nowait()
            except:
                break

    @property
    def provide_data(self):
        return self.data_iter.provide_data

    @property
    def provide_label(self):
        return self.data_iter.provide_label

    @property
    def provide_loss_queue(self):
        return self.data_iter.provide_loss_queue

    @property
    def provide_index_queue(self):
        return self.data_iter.provide_index_queue

    def reset(self):
        self.next_iter = 0
        self.__clear_queue()
        self.data_iter.reset()
        self.prefetch_reset_event.set()
        self.next_reset_event.set()

    def iter_next(self):
        self.next_iter += 1
        if self.next_iter > self.epoch_size:
            self.prefetch_reset_event.clear()
            self.next_reset_event.clear()
            return False
        else:
            return True

    def next(self):
        if self.iter_next():
            if not self.data_iter.update_loss_queue.empty():
                update_loss = self.data_iter.update_loss_queue.get()
                update_index = self.data_iter.update_index_queue.get()
                if self.data_iter.if_sample:
                    self.data_iter.update_prob(update_index, update_loss)

            batch, label, index = self._data_buffer.get(block=True)
            return io.DataBatch(data=[batch], label=[label], index=[index])
        else:
            raise StopIteration

def PrefetchFaceIter(prefetch_process=8, prefetch_process_keep=16, **kwargs):
    data_iter = PrefetchProcessIter(FaceImageIter(**kwargs), prefetch_process, prefetch_process_keep)
    import atexit
    atexit.register(lambda a : a.__del__(), data_iter)
    return data_iter

