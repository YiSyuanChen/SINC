export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=6000
export NODE_RANK=0
export TOKENIZERS_PARALLELISM=true

TASK_NAME=SINC
BATCH_SIZE=16
LOAD_PATH=../Weights/meter/meter_clip16_288_roberta_pretrain.ckpt

if [ $TASK_NAME == "SINC" ]
then
    DATASET=CAPTIONS/concept_classification
    LABEL_SIZE=3926
    TASK=task_finetune_sinc_clip_bert
    RESOLUTION=288
elif [ $TASK_NAME == "VQAv2" ]
then
    DATASET=VQAv2-TAP
    TASK=task_finetune_vqa_clip_bert
    RESOLUTION=288
elif [ $TASK_NAME == "SNLI-VE" ]
then
    DATASET=SNLI-VE-Prompt
    TASK=task_finetune_snli_clip_bert
    RESOLUTION=288
elif [ $TASK_NAME == "NLVR2" ]
then
    DATASET=NLVR2-Prompt
    TASK=task_finetune_nlvr2_clip_bert
    RESOLUTION=288
elif [ $TASK_NAME == "OPEN-MI" ]
then
    DATASET=OPEN-MI-Prompt
    TASK=task_finetune_open_mi_clip_bert
    RESOLUTION=288
fi

##### Train #####
python run.py with \
    $TASK clip16 text_roberta \
    num_gpus=1 num_nodes=1 per_gpu_batchsize=$BATCH_SIZE num_workers=4 \
    data_root=../Datasets/arrows/$DATASET \
    load_path=$LOAD_PATH \
    image_size=$RESOLUTION \
    val_check_interval=0.2 \
    experiment_name=_dev \
    pcls_label_size=$LABEL_SIZE \
    learning_rate=3e-4 \
    max_steps=500000 \
    warmup_steps=4000 \
    freeze_backbone=True \
    use_meta_encoder=True \
    meta_encoder_size=gpt2 \
    meta_save_metric=sum \
    bursty_ratio=0.1 \
    #no_example_baseline=True \
    #example_num=16 \
    #pretrain_scale=large \
    #feature_source=blip2 \
    #only_v_and_l_feats=True \
    #only_vl_feats=True \


##### Test #####
BATCH_SIZE=128
LOAD_PATH=result/finetune_sinc_seed0_from_meter_clip16_288_roberta_pretrain_dev/version_0/checkpoints/epoch=0-step=1404.ckpt

python run.py with \
    $TASK clip16 text_roberta \
    num_gpus=1 num_nodes=1 per_gpu_batchsize=$BATCH_SIZE num_workers=4 \
    data_root=../Datasets/arrows/$DATASET \
    load_path=$LOAD_PATH \
    image_size=$RESOLUTION \
    val_check_interval=1.0 \
    experiment_name=_dev \
    pcls_label_size=$LABEL_SIZE \
    freeze_backbone=True \
    use_meta_encoder=True \
    meta_encoder_size=gpt2 \
    test_only=True \
    vqa_test_with_labels=True \
    #no_example_baseline=True \
    #example_num=16 \
    #feature_source=blip2 \
    #only_v_and_l_feats=True \
    #only_vl_feats=True \
    #select_demonstrations=False \
