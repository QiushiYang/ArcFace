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
import sklearn
import datetime
import numpy as np
import cv2
import copy

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import io
from mxnet import recordio
from mxnet import context
import mxnet.gluon.data.dataloader as dataloader
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
import face_preprocess
import multiprocessing
import threading
import Queue

logger = logging.getLogger()

class FaceImageIter(io.DataIter):

    def __init__(self, batch_size, data_shape,
                 num_samples_per_class = 1,
                 max_samples_per_class = 1,
                 sample_type = 0,
                 repet_flag = 0, repet_num = 1,
                 path_imgrec = None,
                 shuffle=False, aug_list=None, mean = None,
                 rand_mirror = False, cutoff = 0, 
                 data_name='data', label_name='softmax_label', **kwargs):
        super(FaceImageIter, self).__init__()
        assert path_imgrec
        if sample_type == 1 or sample_type == 2:
            assert int(batch_size / num_samples_per_class) * num_samples_per_class == batch_size
        if sample_type == 3:
          assert int(max_samples_per_class / num_samples_per_class) * num_samples_per_class == max_samples_per_class
        if repet_flag:
           assert repet_num > 1

           
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
              self.id_num = {}
              self.seq_identity = range(int(header.label[0]), int(header.label[1]))
              for identity in self.seq_identity:
                s = self.imgrec.read_idx(identity)
                header, _ = recordio.unpack(s)
                a,b = int(header.label[0]), int(header.label[1])
                self.id2range[identity] = (a,b)
                count = b-a

                self.id_num[identity] = b - a
              print('id2range', len(self.id2range))
            else:
              self.imgidx = list(self.imgrec.keys)
            if shuffle:
              if sample_type == 0:
                self.seq = self.imgidx
              elif sample_type == 1 or sample_type == 2:
                self.seq = None
              self.oseq = self.imgidx
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
        self.nbatch = 0
        self.is_init = False
        self.sample_type = sample_type 
        self.num_samples_per_class = num_samples_per_class
        self.repet_flag = repet_flag
        self.repet_num = repet_num
        self.cur_repet = 0
        self.last_data = None
        self.max_samples_per_class = max_samples_per_class
        self.reset()

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        print('call reset()')
        self.cur = 0
        
        if self.repet_flag:
           self.cur_repet = 0
           self.last_data = None
          
        if self.shuffle:
          # random sampling
          if self.sample_type == 0:
            random.shuffle(self.seq)

          # PK sampling, partial samples are used in each epoch
          elif self.sample_type == 1:
            seq_identity = copy.deepcopy(self.seq_identity)
            random.shuffle(seq_identity)

            self.seq = []
            for identity in seq_identity:
              id2range = range(self.id2range[identity][0], self.id2range[identity][1])
              if self.id_num[identity] < self.num_samples_per_class:
                id2range = np.random.choice(id2range, size=self.num_samples_per_class, replace=True)
              else:
                id2range = np.random.choice(id2range, size=self.num_samples_per_class, replace=False)
              self.seq.extend(id2range)
           #random.shuffle(self.seq)
         
          # PK sampling, all samples are used in each epoch
          elif self.sample_type == 2:

            id_seq = []            
            img_seq = []
            self.seq = []
            seq_identity = copy.deepcopy(self.seq_identity)

            id2range = {}
            for identity in seq_identity:
              id2range[identity] = range(self.id2range[identity][0], self.id2range[identity][1])

            while(len(seq_identity) > 0):
              random.shuffle(seq_identity)
              for identity in seq_identity:
                id_seq.append(len(id_seq))

                random.shuffle(id2range[identity])
                if len(id2range[identity]) < self.num_samples_per_class:
                  id_samples = list(np.random.choice(id2range[identity], size=self.num_samples_per_class, replace=True))
                  id2range[identity] = []                 
                else:
                  id_samples = id2range[identity][: self.num_samples_per_class]
                  id2range[identity] = id2range[identity][self.num_samples_per_class :]
                if id2range[identity] == []:
                  seq_identity.remove(identity)
                  del id2range[identity]

                img_seq.append(id_samples)

            random.shuffle(id_seq)
            for i in id_seq:
              self.seq.extend(img_seq[i])

          # PK sampling, partial samples are used in each epoch
          elif self.sample_type == 3:

            id_seq = []            
            img_seq = []
            self.seq = []
            seq_identity = copy.deepcopy(self.seq_identity)

            id2range = {}
            for identity in seq_identity:
              id2range[identity] = range(self.id2range[identity][0], self.id2range[identity][1])
              if len(id2range[identity]) > self.max_samples_per_class:
                id2range[identity] = list(np.random.choice(id2range[identity], size=self.max_samples_per_class, replace=False))

            while(len(seq_identity) > 0):
              random.shuffle(seq_identity)
              for identity in seq_identity:
                id_seq.append(len(id_seq))

                random.shuffle(id2range[identity])
                if len(id2range[identity]) < self.num_samples_per_class:
                  id_samples = list(np.random.choice(id2range[identity], size=self.num_samples_per_class, replace=True))
                  id2range[identity] = []                 
                else:
                  id_samples = id2range[identity][: self.num_samples_per_class]
                  id2range[identity] = id2range[identity][self.num_samples_per_class :]
                if id2range[identity] == []:
                  seq_identity.remove(identity)
                  del id2range[identity]

                img_seq.append(id_samples)

            random.shuffle(id_seq)
            for i in id_seq:
              self.seq.extend(img_seq[i])
         
          #id shuffle only  
          elif self.sample_type == 4:
            for identity in seq_identity:
              id2range = range(self.id2range[identity][0], self.id2range[identity][1])
              self.seq.extend(id2range)

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
              return label, img, None, None
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
              return label, img, None, None
            else:
              label, fname, bbox, landmark = self.imglist[idx]
              return label, self.read_image(fname), bbox, landmark
        else:
            s = self.imgrec.read()
            if s is None:
                raise StopIteration
            header, img = recordio.unpack(s)
            return header.label, img, None, None

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

    def next(self, lock=None, imgrec=None, index=None):
        """Returns the next batch of data."""
        batch_size = self.batch_size
        c, h, w = self.data_shape
        batch_data = nd.empty((batch_size, c, h, w))
        if self.provide_label is not None:
            batch_label = nd.empty(self.provide_label[0][1])
        if index is None:
            index = random.sample(range(0, len(self.seq)), batch_size)
        i = 0
        try:
            while i < batch_size:
                label, s, bbox, landmark = self.next_sample(index[i], lock, imgrec)
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
                #print('aa',data[0].shape)
                #data = self.augmentation_transform(data)
                #print('bb',data[0].shape)
                for datum in data:
                    assert i < batch_size, 'Batch size must be multiples of augmenter output length'
                    #print(datum.shape)
                    batch_data[i][:] = self.postprocess_data(datum)
                    batch_label[i][:] = label
                    i += 1
        except StopIteration:
            if i<batch_size:
                raise StopIteration

        return io.DataBatch([batch_data], [batch_label], batch_size - i)

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

        if hasattr(self.data_iter, 'epoch_size'):
            self.epoch_size = self.data_iter.epoch_size
            if self.data_iter.epoch_size is None:
                self.epoch_size = int(self.data_iter.num_samples/self.batch_size)
        else:
            self.epoch_size = int(self.data_iter.num_samples/self.batch_size)

        self.next_iter = 0
        self.prefetch_process = prefetch_process

        self._data_queue = dataloader.Queue(maxsize=capacity)
        self._data_buffer = Queue.Queue(maxsize=capacity*2)
        self._index_queue = multiprocessing.Queue()

        self.prefetch_reset_event = multiprocessing.Event()
        self.epoch_end_event = multiprocessing.Event()
        self.next_reset_event = threading.Event()

        self.lock = multiprocessing.Lock()
        self.imgrec = self.data_iter.imgrec

        def prefetch_func(data_queue, event, end_event):
            while True:
                if event.is_set() and (not end_event.is_set()):
                    index = []
                    i = 0
                    while i < self.batch_size:
                        try:
                            index.append(self._index_queue.get())
                            i += 1
                        except:
                            end_event.set()
                    if i == self.batch_size:
                        next_data = self.data_iter.next(self.lock, self.imgrec, index)
                        data_queue.put((dataloader.default_mp_batchify_fn(next_data.data[0]),
                                       (dataloader.default_mp_batchify_fn(next_data.label[0]))))

        def next_func(data_queue, event):
            while True:
                if event.is_set():
                    batch, label = data_queue.get(block=True)
                    batch = dataloader._as_in_context(batch, context.cpu())
                    label = dataloader._as_in_context(label, context.cpu())
                    self._data_buffer.put((batch, label))

        # producer next
        self.produce_lst = []
        for ith in range(prefetch_process):
            p_process = multiprocessing.Process(target=prefetch_func,
                                                args=(self._data_queue, self.prefetch_reset_event,
                                                      self.epoch_end_event))
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

    def reset(self):
        self.next_iter = 0
        self.__clear_queue()
        self.data_iter.reset()

        assert self._index_queue.empty()
        seq_index = range(0, len(self.data_iter.seq))
        random.shuffle(seq_index)
        for index in range(0, len(self.data_iter.seq)):
            self._index_queue.put(seq_index[index])

        self.prefetch_reset_event.set()
        self.next_reset_event.set()
        self.epoch_end_event.clear()

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
            batch, label = self._data_buffer.get(block=True)
            return io.DataBatch(data=[batch], label=[label])
        else:
            raise StopIteration

def PrefetchFaceIter(prefetch_process=8, prefetch_process_keep=16, **kwargs):
    data_iter = PrefetchProcessIter(FaceImageIter(**kwargs), prefetch_process, prefetch_process_keep)
    import atexit
    atexit.register(lambda a : a.__del__(), data_iter)
    return data_iter

