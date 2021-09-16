#!/bin/bash
# process named arguments
for ARGUMENT in "$@"
do

    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   

    case "$KEY" in
            GPUS)     GPUS=${VALUE} ;;
            BG) BG=${VALUE} ;;
            MODEL)     MODEL=${VALUE} ;;
            *)   
    esac    
done

GPUS=${GPUS:-0}
BG=${BG:-0}
MODEL=${MODEL:-"transformer"}

# since nohup creates its own child process to run train function,
# we have export them as env variables

export EX_POSTFIX=$(date +%T | sed "s/:/-/g" | echo "-$(cat -)")
export ARCH=$MODEL"_wmt_en_de"
export DATA=data-bin/wmt16_en_de_bpe32k
export CKPT_DIR="checkpoints/""${ARCH//_/-}""$EX_POSTFIX"
export UPDATE_FREQ=$(( 8 / (1 + (${#GPUS} / 2)) ))
mkdir -p $CKPT_DIR

train() {
    fairseq-train $DATA \
            --arch $ARCH --share-all-embeddings \
            --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
            --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
            --lr 0.0007 --stop-min-lr 1e-09 --dropout 0.1\
            --tensorboard-logdir $CKPT_DIR --seed 2\
            --best-checkpoint-metric ppl\
            --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0\
            --max-tokens  4096 --save-dir $CKPT_DIR\
            --update-freq $UPDATE_FREQ --no-progress-bar --log-interval 50\
            --save-interval 10  --keep-last-epochs 1 --max-update 275000\
            --save-interval-updates  1000 --keep-interval-updates 15 --ddp-backend=no_c10d
}
export -f train 

# nohup sh -c '(date > fmt1 2>&1 ; sleep 2; date > fmt2)' &

if [ $BG == 1 ];
then
    CUDA_VISIBLE_DEVICES=$GPUS nohup bash -c "train" > $CKPT_DIR/train.output 2>&1 &
else
    CUDA_VISIBLE_DEVICES=$GPUS train
fi
