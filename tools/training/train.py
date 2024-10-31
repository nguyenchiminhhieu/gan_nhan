from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer

config = Cfg.load_config_from_name('vgg_seq2seq')

dataset_params = {
    'name':'hw',
    'data_root':'./Dataset_VB/',
    'train_annotation':'train_annotation.txt',
    'valid_annotation':'test_annotation.txt'
}

params = {
         'print_every':200,
         'valid_every':15*200,
          'iters':120,
          'checkpoint':'./checkpoint/seq2seqocr_checkpoint.pth',    
          'export':'./weights/seq2seqocr.pth',
          'metrics': 60
         }

# Ensure all required parameters are set
config['trainer'].update(params)
config['dataset'].update(dataset_params)
config['device'] = 'cuda:0'
config['dataloader']['num_workers'] = 0

trainer = Trainer(config, pretrained=True)
trainer.train()
trainer.config.save('config.yml')
trainer.visualize_dataset()