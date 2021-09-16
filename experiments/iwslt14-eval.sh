# process named arguments
#!/bin/bash

usage() { echo "Usage: $0 [-p <string>]" 1>&2; exit 1; }

while getopts ":p:g:" o; do
    case "${o}" in
        p)
            CKPT_DIR=${OPTARG}
            ;;
        g)
            GPUS=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

if [ -z "${CKPT_DIR}" ]; then
    usage
fi

# The shift will remove the first argument, renaming the rest ($2 becomes $1, and so on).
GPUS=${GPUS:-0}
CKPT_DIR=${CKPT_DIR:-''}
DATA=data-bin/iwslt14.tokenized.de-en


grep ".*valid on 'valid' subset.*" $CKPT_DIR/train.output > $CKPT_DIR/train-log.output 2>&1

CUDA_VISIBLE_DEVICES=$GPUS fairseq-generate $DATA \
    --path $CKPT_DIR/checkpoint_best.pt \
    --batch-size 128 --remove-bpe --beam 5 > $CKPT_DIR/generate.out 2>&1

# CUDA_VISIBLE_DEVICES=$GPUS fairseq-generate $DATA \
#     --path $CKPT_DIR/checkpoint_best.pt \
#     --batch-size 128 --remove-bpe --beam 4 --lenpen 0.6 \
#     --max-len-a 1 --max-len-b 50 > $CKPT_DIR/generate.out 2>&1


# "compound split" tokenized BLEU
bash scripts/compound_split_bleu.sh $CKPT_DIR/generate.out > $CKPT_DIR/eval.output


# detokenized BLEU with sacrebleu 
# bash scripts/sacrebleu.sh wmt14/full en de gen.out

# Put compounds in ATAT format (comparable to papers like GNMT, ConvS2S).
# See https://nlp.stanford.edu/projects/nmt/ :
# 'Also, for historical reasons, we split compound words, e.g.,
#    "rich-text format" --> rich ##AT##-##AT## text format."'
# grep ^T $CKPT_DIR/generate.out | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $CKPT_DIR/generate.ref
# grep ^H $CKPT_DIR/generate.out | cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $CKPT_DIR/generate.sys
# fairseq-score --sys $CKPT_DIR/generate.sys --ref $CKPT_DIR/generate.ref > $CKPT_DIR/eval.output