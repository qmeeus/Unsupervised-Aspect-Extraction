#!/bin/sh

#exec tensorboard --logdir ./logs &
cd code

domain=restaurant

if [ "$@" == "train" ]; then
    echo "Training"
    exec python3 train.py \
        --emb ../preprocessed_data/$domain/w2v_embedding \
        --domain $domain \
        -o output_dir
else
    echo "Evaluate"
    exec python3 evaluation.py \
        --domain $domain \
        -o output_dir
fi

