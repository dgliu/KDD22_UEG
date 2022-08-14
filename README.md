## KDD22_UEG

* * *

Experiments codes for the paper:

Dugang Liu, Mingkai He, Jinwei Luo, Jiangxu Lin, Meng Wang, Xiaolian Zhang, Weike Pan and Zhong Ming. User-Event Graph Embedding Learning for Context-Aware Recommendation. To appear in SIGKDD '22.

**Please cite our SIGKDD '22 paper if you use our codes. Thanks!**

* * *

## Requirement
python==3.6.9

tensorflow==1.15.3

faiss-gpu==1.7.2

* * *

## Usage

Note that our implementation is based on GCM ([Link](https://github.com/wujcan/GCM)), and you may need to follow its instructions to compile the evaluator first.

For UEG-EL, a Yelp-OH based example is,

`python UEG_EL.py --dataset Yelp-OH --num_gcn_layers 2 --reg 1e-1 --decoder_type FM --adj_norm_type ls --num_negatives 4 --test_batch_size 50 --pretrain 0 --save_flag 1 --finetune 0 --grouping 0`

For UEG-EL-V, a Yelp-OH based example is,

`python UEG_EL.py --dataset Yelp-OH --num_gcn_layers 2 --reg 1e-1 --decoder_type FM --adj_norm_type ls --num_negatives 4 --test_batch_size 50 --pretrain 1 --pretrain_layer 2 --pretrain_reg 1e-1 --save_flag 0 --finetune 0 --grouping 1 --keep_rate 0.9`

`python UEG_EL.py --dataset Yelp-OH --num_gcn_layers 2 --reg 1e-1 --decoder_type FM --adj_norm_type ls --num_negatives 4 --test_batch_size 50 --pretrain 1 --pretrain_layer 2 --pretrain_reg 1e-1 --save_flag 0 --finetune 1 --grouping 0 --keep_rate 0.9 --lr 0.0001`

* * *

If you have any issues or ideas, feel free to contact us ([dugang.ldg@gmail.com](mailto:dugang.ldg@gmail.com)).
