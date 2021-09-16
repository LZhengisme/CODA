#!/bin/bash


#!/bin/bash

# process named arguments
for ARGUMENT in "$@"
do

    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   

    case "$KEY" in
            GPUS)     GPUS=${VALUE} ;;
            BG) BG=${VALUE} ;;
            MODEL) MODEL=${VALUE} ;;
            HEAD) HEAD=${VALUE} ;;
            *)   
    esac    
done

GPUS=${GPUS:-0}
BG=${BG:-0}
MODEL=${MODEL:-"coda"}
# since nohup creates its own child process to run train function,
# we have export them as env variables

export EX_POSTFIX=$(date +%T | sed "s/:/-/g" | echo "-$(cat -)")
export ARCH=$MODEL"_iwslt14_de_en"
export DATA=data-bin/iwslt14.tokenized.de-en
export HEAD=${HEAD:-4}
export CKPT_DIR="checkpoints/""${ARCH//_/-}""$EX_POSTFIX"
mkdir -p $CKPT_DIR
train() {

    fairseq-train $DATA \
    --arch $ARCH --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --activation-dropout 0.1 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --tensorboard-logdir $CKPT_DIR --seed 1\
    --max-tokens 2048 --max-epoch 100 --update-freq 2\
    --best-checkpoint-metric loss --encoder-attention-heads $HEAD --decoder-attention-heads $HEAD\
    --no-progress-bar --log-interval 50\
    --save-dir $CKPT_DIR --save-interval 1  --keep-last-epochs 3 \
    --ddp-backend=no_c10d \
    # --validate-interval-updates 1 \
    # --save-interval-updates 1000 --keep-interval-updates 15
    #--max-update 200000\
    # --eval-bleu \
    # --eval-bleu-args '{"beam": 4, "max_len_a": 0.6}' \
    # --eval-bleu-detok moses \
    # --eval-bleu-remove-bpe \
    # --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
}
export -f train 


# nohup sh -c '(date > fmt1 2>&1 ; sleep 2; date > fmt2)' &

if [ $BG == 1 ];
then
    CUDA_VISIBLE_DEVICES=$GPUS nohup bash -c "train" > $CKPT_DIR/train.output 2>&1 &
else
    CUDA_VISIBLE_DEVICES=$GPUS train
fi