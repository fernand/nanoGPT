python train.py config/train_shakespeare_char.py\
    --block_size=64 --batch_size=64\
    --n_layer=4 --n_head=4 --n_embd=128\
    --learning_rate=1e-3 --dropout=0.0\
    --log_interval=500 --eval_iters=200\
    --max_iters=7000 --lr_decay_iters=7000 \
