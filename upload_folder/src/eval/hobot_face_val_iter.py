__author__ = 'yu01.wang','mengjia.yan'
import mxnet as mx

def get_val_iter2(args,kv,mean): #args , kv
    data_shape = (3, 112, 96)
    dataset_path = args.val_data_dir
    batch_size = args.batch_size#/args.split_num
    hr_928_val = mx.io.ImageRecordIter(
        #path_imgrec=dataset_path + 'metric_valid_gray_equhist_rec/merge_gray_equhist.rec',
        path_imgrec=dataset_path + '928faces_112x96.rec',#'metric_valid_gray_equhist_rec/merge_gray_equhist.rec',
        #mean_img=args.data_dir + "mean.bin",
        data_shape=data_shape,
        batch_size=batch_size, #if args.split_num is None else args.batch_size / args.split_num,
        round_batch=False,
        rand_crop=False,
        rand_mirror=False,
        #num_parts=kv.num_workers,
        #part_index=kv.rank,
        #mean_r=mean[0], mean_g=mean[1], mean_b=mean[2],
        scale=1.0,
        prefetch_buffer=4,
        prefetch_buffer_keep=2,
        use_equhist=0,
    )
    hr_hobot_val = mx.io.ImageRecordIter(
        #path_imgrec=dataset_path + 'metric_valid_gray_equhist_rec/merge_gray_equhist.rec',
        path_imgrec=dataset_path + 'val238_112x96.rec',#'metric_valid_gray_equhist_rec/merge_gray_equhist.rec',
        #mean_img=args.data_dir + "mean.bin",
        data_shape=data_shape,
        batch_size=batch_size, #if args.split_num is None else args.batch_size / args.split_num,
        round_batch=False,
        rand_crop=False,
        rand_mirror=False,
        #num_parts=kv.num_workers,
        #part_index=kv.rank,
        preprocess_thread=8,
        #mean_r=mean[0], mean_g=mean[1], mean_b=mean[2],
        scale=1.0,
        prefetch_buffer=4,
        prefetch_buffer_keep=2,
        use_equhist=0,
    )
    #dataset_path= HDFS + 'user/xin.wang/data/face/face_affine/'
    id_photo_val = mx.io.ImageRecordIter(
        #path_imgrec=dataset_path + 'metric_valid_gray_equhist_rec/merge_gray_equhist.rec',
        path_imgrec=dataset_path + 'id_test_0812.rec',#'metric_valid_gray_equhist_rec/merge_gray_equhist.rec',
        #mean_img=args.data_dir + "mean.bin",
        data_shape=data_shape,
        batch_size=batch_size, #if args.split_num is None else args.batch_size / args.split_num,
        round_batch=False,
        rand_crop=False,
        rand_mirror=False,
        #num_parts=kv.num_workers,
        #part_index=kv.rank,
        #mean_r=mean[0], mean_g=mean[1], mean_b=mean[2],
        scale=1.0,
        prefetch_buffer=4,
        prefetch_buffer_keep=2,
        use_equhist=0,
        preprocess_threads=8,
    )

    val_life = mx.io.ImageRecordIter(
        #path_imgrec=dataset_path + 'metric_valid_gray_equhist_rec/merge_gray_equhist.rec',
        path_imgrec=dataset_path + 'valLife_112x96.rec',#'metric_valid_gray_equhist_rec/merge_gray_equhist.rec',
        #mean_img=args.data_dir + "mean.bin",
        data_shape=data_shape,
        batch_size=batch_size, #if args.split_num is None else args.batch_size / args.split_num,
        round_batch=False,
        rand_crop=False,
        rand_mirror=False,
        #num_parts=kv.num_workers,
        #part_index=kv.rank,
        #mean_r=mean[0], mean_g=mean[1], mean_b=mean[2],
        scale=1.0,
        prefetch_buffer=4,
        prefetch_buffer_keep=2,
        use_equhist=0,
        preprocess_threads=8,
    )

    val_sets = [hr_hobot_val, hr_928_val, id_photo_val, val_life]  #,, lfw_val, id_photo_val
    return val_sets#[(kv.num_workers-kv.rank-1):len(val_sets):kv.num_workers]
    #return val_sets
    #return val_sets[(kv.num_workers-kv.rank-1):len(val_sets):kv.num_workers]
