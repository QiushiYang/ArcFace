# 使用说明

第一步，把`config.sh`打开；第二步，修改实验参数; 第三步，运行`submit.sh`。

## 实验参数及`config.sh`

`config.sh`中可修改的参数在第54行之前，即，


```sh

#!/bin/bash
# config.sh

export JOB_NAME="experiment-merge-modelparallel-dataparallel-codes"
export CLUSTER_NAME="aliyun"  # choose cluster among "mos", "ksyun" and "aliyun"
export TRAIN_SET="GLINT_ABOVE_10"  # choose training set
...
...
export IMG_SIZE='112x96'
export DATAITER='ImageRecordIter' # choose from FaceImageIter and ImageRecordIter
export PER_BATCHSIZE=128
export KVSTORE="local"
export WEIGHT_DECAY=0.00005
if [ "$TRAIN_MODE" == "model_parallel" ]; then
    export LR=1
    export LR_FACTOR_EPOCH=5
    export LR_FACTOR=0.2
    export EPOCH_SIZE=3000
    export EVAL_FIRST=0
elif [ "$TRAIN_MODE" == "data_parallel" ]; then
    export LR=0.1
    export NUM_SAMPLES_PER_CLASS=0
fi

```

### 注意事项

- `CLUSTER_NAME`为集群名， 必须为目前支持的三个集群：美团云“mos”，阿里云“aliyun”，金山云“ksyun”。
- 当使用美团云时，任务名`JOB_NAME`在`config.sh`中修改即可。但如果是阿里云或金山云，需要在`job.yaml`中修改。(**TODO**)
- 目前仍没有设置计算资源的参数，需要手动设置。美团云在`submit.sh`设置，阿里云和金山云在`job.yaml`中设置。（**TODO**）
- 当使用阿里云或金山云时，需要根据训练方式（模型并行“model_parallel”，数据并行“data_parallel”）修改`job.yaml`中的`RUN_SCRIPTS`参数。(**TODO**)
    - 模型并行：`${WORKING_PATH}/job_model_parallel.sh`
    - 数据并行：`${WORKING_PATH}/job_data_parallel.sh`
- **集群名，训练集，训练方式，DataIter，等参数的选择不要拼写错误，bash脚本不会报错。。。**

### 文件路径

- 验证集：目前共用四个验证集：_val928, val238, valLife, wanren_。路径为`$VAL_DATA_DIR`, i.e. `${HDFS}'user/mengjia.yan/valSet/'${IMG_SIZE}'/imagerecord/'`
- 训练集：目前共用五个训练集：*GLINT_ABOVE_10, GLINT_ABOVE_5, 9.5_above10, 9.8_above5, 11.5_above10*。 路径为`$DATA_REC_PATH`。
- `$PREFIX`可自己设置，最好包含`$JOB_NAME`信息。
