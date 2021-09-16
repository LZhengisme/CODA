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
# The shift will remove the first argument, renaming the rest ($2 becomes $1, and so on).
shift $((OPTIND-1))

if [ -z "${CKPT_DIR}" ]; then
    usage
fi


GPUS=${GPUS:-0}
CKPT_DIR=${CKPT_DIR:-''}
DATA=data-bin/wmt16_en_de_bpe32k

if [ ! -f "$CKPT_DIR/checkpoint.avg10.pt" ]; then
python scripts/average_checkpoints.py \
--inputs $CKPT_DIR \
--num-update-checkpoints 10 \
--output $CKPT_DIR/checkpoint.avg10.pt
fi


CUDA_VISIBLE_DEVICES=$GPUS fairseq-generate \
$DATA \
--path $CKPT_DIR/checkpoint.avg10.pt \
--beam 4 --lenpen 0.6 --remove-bpe > $CKPT_DIR/gen.out    


# # "compound split" tokenized BLEU
bash scripts/compound_split_bleu.sh $CKPT_DIR/gen.out

# detokenized BLEU with sacrebleu 
# bash scripts/sacrebleu.sh wmt14/full en de gen.out