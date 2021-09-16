# CODA: Cascaded Head-colliding Attention

This repo contains code for our ACL'2021 paper: [Cascaded Head-colliding Attention](https://arxiv.org/pdf/2105.14850.pdf). We use [Fairseq](https://github.com/pytorch/fairseq/) codebase for our machine translation and language modeling experiments.

## Installation
Please ensure that:
- PyTorch version >= 1.5.0
- Python version >= 3.6

To install our codebase:
``` bash
git clone https://github.com/LZhengisme/CODA
cd CODA
pip install --editable ./
```

## Data preparation

Please refer to the official repository in [Fairseq](https://github.com/pytorch/fairseq) for preparation details.

- For IWSLT-14 de-en dataset, we follow [here](https://github.com/pytorch/fairseq/tree/master/examples/translation#iwslt14-german-to-english-transformer) to pre-process and binarize data.
- For WMT-14 en-de dataset, follow [here](https://github.com/pytorch/fairseq/blob/master/examples/scaling_nmt/README.md#training-a-new-model-on-wmt16-en-de) to prepare for data; note that you need to first download the preprocessed data provided by Google.
- For Wikitext-103, follow [here](https://github.com/pytorch/fairseq/blob/master/examples/language_model/README.md#1-preprocess-the-data) to pre-process and binarize data.

## Training and Evaluation
We provide a series of scripts (located at `experiments` folder) for training or evaluating models on both machine translation and language modeling. Further details and hyper-parameter settings can be found either in these scripts or our [paper](https://arxiv.org/pdf/2105.14850.pdf).

Taking **IWSLT-14 dataset** as an example, the following command would train our CODA model with default settings: 
``` bash
bash experiments/iwslt14-train.sh GPUS=0,1,2,3 BG=1
```
- Assuming there are 4 GPUs available, and
- `BG=1` indicates that the script will run in background (via `nohup`).
All of checkpoints and the training log will be saved in `checkpoints/coda-iwslt14-de-en-<current time>` folder.

To evaluate the trained model:
``` bash
bash experiments/iwslt14-eval.sh -p checkpoints/coda-iwslt14-de-en-<current time> -g 0
```
- `-p` must be specified as the path where your checkpoints are saved, and
- `-g` (optional) is used for selection GPUs (default selecting GPU 0).

If everything goes smoothly, you should get a decent BLEU score (~35.7) after training 100 epochs.

## Citation
Please cite our paper as:
``` bibtex
@inproceedings{zheng-etal-2021-cascaded,
    title = "Cascaded Head-colliding Attention",
    author = "Zheng, Lin  and
      Wu, Zhiyong  and
      Kong, Lingpeng",
    booktitle = "ACL-IJCNLP",
    year = "2021",
    url = "https://aclanthology.org/2021.acl-long.45",
    doi = "10.18653/v1/2021.acl-long.45",
}
```

Our code is based on [Fairseq](https://github.com/pytorch/fairseq/). To cite Fairseq:
``` bibtex
@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```