data_path = './data'
bert_path = 'D:/Models/bert_pretrain/chinese_L-12_H-768_A-12/'
checkpoint_path = 'D:/Project/Machine_translation/School_Experiment_v3/'
predict_file = 'out.txt'
epochs = 100

max_seq_len = 128

en_vocab_size = 2 ** 13
batch_size_per_replica = 150

beam_size = 4
lp_alpha = 0.6  # length penalty in beam search

d_model = 512
num_layers = 6
d_ff = 2048
num_heads = 8

dec_config = {
    'num_layers': num_layers,
    'd_model': d_model,
    'num_heads': num_heads,
    'd_ff': d_ff,
    'target_vocab_size': en_vocab_size + 2
}

bert_config_file = bert_path + "bert_config.json"
bert_ckpt_file = bert_path + "bert_model.ckpt"
