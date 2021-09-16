#!/bin/bash

# process named arguments
err_msg() { echo "Invalid arguments" 1>&2; exit 1; }

while getopts ":p:g:m:t:e:b:" o; do
    case "${o}" in
        p)
            CKPT_DIR=${OPTARG}
            ;;
        g)
            GPUS=${OPTARG}
            ;;
        m)
            MODEL=${OPTARG}
            ;;
        t)
            TRAIN=${OPTARG}
            ;;
        e)
            EVAL=${OPTARG}
            ;;
        b)
            BG=${OPTARG}
            ;;
        *)
            err_msg
            ;;
    esac
done
shift $((OPTIND-1))
GPUS=${GPUS:-0}
TRAIN=${TRAIN:-0}
EVAL=${EVAL:-0}
MODEL=${MODEL:-"transformer"}
BG=${BG:-0}
PREPROC=0

if [ $TRAIN == $EVAL ]; then
    echo "TRAIN and EVAL cannot be the same. Exit now."
    exit 1
fi
if [ $PREPROC == 1 ]
then
    TEXT=examples/language_model/wikitext-103
    fairseq-preprocess \
        --only-source \
        --trainpref $TEXT/wiki.train.tokens \
        --validpref $TEXT/wiki.valid.tokens \
        --testpref $TEXT/wiki.test.tokens \
        --destdir data-bin/wikitext-103 \
        --workers 20
fi

if [ $TRAIN == 1 ]
then
    export EX_POSTFIX=$(date +%T | sed "s/:/-/g" | echo "-$(cat -)")
    export ARCH=$MODEL"_lm_wiki103"
    export DATA=data-bin/wikitext-103
    export CKPT_DIR="checkpoints/""${ARCH//_/-}""$EX_POSTFIX"
    # as $GPUS is of the form ''1,2,3,4,...''
    # ${#GPUS} / 2 + 1 will yield the correct number of GPUS.
    export UPDATE_FREQ=$(( 16 / (${#GPUS} / 2 + 1) ))
    mkdir -p $CKPT_DIR
    train() {
        fairseq-train --task language_modeling $DATA \
            --arch $ARCH --no-progress-bar --log-interval 50 \
            --max-update 286000 --lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 \
            --warmup-updates 16000 --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --optimizer nag --min-lr 0.0001 --clip-norm 0.1 \
            --criterion adaptive_loss --max-tokens 4096 --update-freq $UPDATE_FREQ --tokens-per-sample 512 --seed 1 \
            --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d \
            --save-dir $CKPT_DIR --save-interval 10 --keep-last-epochs 2
        # --validate-interval-updates 1 \
        # --save-interval-updates 1000 --keep-interval-updates 15
    }
    export -f train 
    if [ $BG == 1 ]
    then
        CUDA_VISIBLE_DEVICES=$GPUS nohup bash -c "train" > $CKPT_DIR/train.output 2>&1 &
    else
        CUDA_VISIBLE_DEVICES=$GPUS train
    fi
fi

if [ $EVAL == 1 ]
then
    
    if [ -z "${CKPT_DIR}" ]; then
        err_msg
    fi
    DATA=data-bin/wikitext-103
    echo "Testing PPL" >> $CKPT_DIR/eval-lm.log
    CUDA_VISIBLE_DEVICES=$GPUS fairseq-eval-lm $DATA \
        --path $CKPT_DIR/checkpoint_best.pt \
        --batch-size 8 --gen-subset test\
        --tokens-per-sample 512 \
        --context-window 480 >> $CKPT_DIR/eval-lm.log 2>&1

    echo "Validation PPL" >> $CKPT_DIR/eval-lm.log
    CUDA_VISIBLE_DEVICES=$GPUS fairseq-eval-lm $DATA \
        --path $CKPT_DIR/checkpoint_best.pt \
        --batch-size 8 --gen-subset valid\
        --tokens-per-sample 512 \
        --context-window 480 >> $CKPT_DIR/eval-lm.log 2>&1
fi
