from tool.config import Cfg
from tool.translate import build_model, process_input, translate
import torch


config = Cfg.load_config_from_file('./config/vgg-seq2seq.yml')
config['cnn']['pretrained']=False
config['device'] = 'cpu'
model, vocab = build_model(config)

# load weight
weight_path = './weight/seq2seqocr.pth'
model.load_state_dict(torch.load(weight_path, map_location=torch.device(config['device'])))
model = model.eval() 

# CNN part
def convert_cnn_part(img, save_path, model, max_seq_length = 128, sos_token = 1, eos_token = 2): 
    with torch.no_grad(): 
        src = model.cnn(img)
        torch.onnx.export(model.cnn, 
                          img, 
                          save_path, 
                          export_params = True, 
                          opset_version = 12, 
                          do_constant_folding = True, 
                          verbose = True, 
                          input_names = ['img'], 
                          output_names = ['output'], 
                          dynamic_axes = {'img': {3: 'lenght'}, 'output': {0: 'channel'}})
     
    return src

# Encoder part
def convert_encoder_part(model, src, save_path): 
    encoder_outputs, hidden = model.transformer.encoder(src) 
    torch.onnx.export(model.transformer.encoder, 
                      src, 
                      save_path, 
                      export_params = True, 
                      opset_version = 11, 
                      do_constant_folding = True, 
                      input_names  = ['src'], 
                      output_names = ['encoder_outputs', 'hidden'], 
                      dynamic_axes = {'src' : {0 : "channel_input"}, 
                                      'encoder_outputs' : {0 : 'channel_output'}}) 
    return hidden, encoder_outputs

# Decoder part
def convert_decoder_part(model, tgt, hidden, encoder_outputs, save_path):
    tgt = tgt[-1]
    print(tgt)
    torch.onnx.export(  model.transformer.decoder,
                        (tgt, hidden, encoder_outputs),
                        save_path,
                        export_params = True,
                        opset_version = 11,
                        do_constant_folding = True,
                        input_names = ['tgt', 'hidden', 'encoder_outputs'],
                        output_names = ['output', 'hidden_out', 'last'],
                        dynamic_axes = {'encoder_outputs' : {0 : "channel_input"}, 
                                        'last' : {0 : 'channel_output'}})


# Export model
img = torch.rand(1, 3, 32, 475)   # input tensor of torch model (N x C x H x W)
src = convert_cnn_part(img, './weight/cnn.onnx', model)

hidden, encoder_outputs = convert_encoder_part(model, src, './weight/encoder.onnx')

device = img.device
tgt = torch.LongTensor([[1] * len(img)]).to(device)
convert_decoder_part(model, tgt, hidden, encoder_outputs, './weight/decoder.onnx')