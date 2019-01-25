import operator
import mxnet as mx
import numpy as np

np.set_printoptions(suppress=True)

batchSize = 128

idx_dir = '/mnt/data-1/data/qiushi.yang/tmp/data/11.5_above10_rec/9.8_above10_align_96_all.idx'
data_dir = '/mnt/data-1/data/qiushi.yang/tmp/data/11.5_above10_rec/9.8_above10_align_96_all.rec'

imgrec = mx.recordio.MXIndexedRecordIO(idx_dir, data_dir, 'r')
imgrec.reset()

s = imgrec.read_idx(0)
header, _ = mx.recordio.unpack(s)
header0 = (int(header.label[0]), int(header.label[1]))

id2range = {}
seq_identity = range(int(header.label[0]), int(header.label[1]))

for identity in seq_identity:
    s = imgrec.read_idx(identity)
    header, _ = mx.recordio.unpack(s)
    a,b = int(header.label[0]), int(header.label[1])
    id2range[identity] = b-a

# sorted
sorted_dict = sorted(id2range.items(), key=operator.itemgetter(1), reverse=True) #[(ID,number), ... ]
np.save('deepInsight_ms1m_ID_number.npy', sorted_dict)

#frequency statistics
cls_num_stat = {}
for imgNum in sorted_dict:
    key = imgNum[1]
    if key not in cls_num_stat:
        cls_num_stat[key] = 1
    else:
        cls_num_stat[key] += 1
np.save('deepInsight_ms1m_Freq_Dist.npy', cls_num_stat) #[(Frequence, Distribution), ... ]

SegList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 21, 31, 51, 101, 201, 501, 1001, 100001] #[l,r)
totalID = np.zeros(len(SegList) - 1)
totalImage = np.zeros(len(SegList) - 1)
for key in cls_num_stat:
    key = int(key)
    for i in range(len(SegList) - 1):
        if(key >= SegList[i] and key < SegList[i+1]):
            totalID[i] += cls_num_stat[key]
            totalImage[i] += key * cls_num_stat[key]
            break

ID_Number = np.sum(totalID)
Image_Number = np.sum(totalImage)

ratio_ID = totalID / ID_Number
ratio_Image = totalImage / Image_Number

sumID = np.cumsum(totalID)
sumImage = np.cumsum(totalImage)

sumID_ratio = np.cumsum(ratio_ID)
sumImage_ratio = np.cumsum(ratio_Image)

print totalID
print totalImage
print ratio_ID
print ratio_Image
print sumID
print sumImage
print sumID_ratio
print sumImage_ratio
print ID_Number
print Image_Number

