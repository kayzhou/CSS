# sample
nohup  python -u  sample.py \
 --data=stance_data\
 --job=10 \
 --item_max_length=50 \
 --user_max_length=50 \
 --k_hop=3 \
 >./jup_0730 &

# train
nohup  python -u  main.py \
 --data=sample_data\
 --gpu=1 \
 --epoch=20 \
 --hidden_size=128 \
 --batchSize=32 \
 --user_long=orgat \
 --user_short=att \
 --item_long=orgat \
 --item_short=att \
 --user_update=rnn \
 --item_update=rnn \
 --lr=0.001 \
 --l2=0.0001 \
 --layer_num=3 \
 --item_max_length=50 \
 --user_max_length=50 \
 --attn_drop=0.3 \
 --feat_drop=0.3 \
 --record \
 --val\
 --jioaohu=10\
 --save_path=Model\
 >./jup &