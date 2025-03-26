params = dict()

params['input_steps'] = 12
params['pred_steps'] = 12

#params['pretrained'] = '/home/yztang/encoder_with_transformer/Exp/eTrans_V2-use_conv-True-use_station_embedding-True-use_kan-True-N_S-2/best-199.pth.tar'

params['loss'] = 'mse'
params['lamda'] = 0.1

params['batch_size'] = 128
params['GRAD_ACC_STEPS'] = 1
params['num_workers'] = 0

# 学习率设置
params['learning_rate'] = 1e-2
params['momentum'] = 0.9
params['weight_decay'] = 1e-5
params['step'] = 50

params['epoch_num'] = 100
params['save_path'] = 'Exp'

params['model'] = 'MutilGNN'
# params['N_S'] = 2
# params['patch_size'] = 3

params['gpu'] = '0'
# params['use_con'] = True
# params['use_station_embedding'] = True
# params['use_kan'] = True