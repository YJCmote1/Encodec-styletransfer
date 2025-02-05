import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from .base_model import BaseModel
#from base_model import BaseModel
from . import networks
import typing as tp
EncodedFrame = tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]

class AdaAttNModel(BaseModel):
    def __init__(self,opt):
        BaseModel.__init__(self, opt)
        self.max_sample = 64 * 64  # 控制注意力计算的采样规模
        self.seed = 6666
        
        # 适配 encodec 编码的音频输入
        target_bandwidths = [3, 6, 12, 24]
        self.c = None
        self.s = None
        if opt.skip_connection_3:
            adaattn_3 = networks.AdaAttN(in_planes=256, key_planes=256 + 128 + 64 if opt.shallow_layer else 256,
                                              max_sample=self.max_sample)
            self.net_adaattn_3 = networks.init_net(adaattn_3, opt.init_type, opt.init_gain, opt.gpu_ids)
            self.model_names.append('adaattn_3')
            parameters.append(self.net_adaattn_3.parameters())
        self.encodec_model = networks.AudioEncoder._get_model(target_bandwidths)
        self.net_transformer = networks.init_net(
            networks.Transformer(in_planes=opt.k, key_planes=opt.k),
            opt.init_type, opt.init_gain, opt.gpu_ids
        )
        
        self.model_names = ['decoder', 'transformer']
        self.visual_names = ['c', 'cs', 's']
        
        parameters = [
            self.net_transformer.parameters(),
            #self.audio_decoder.parameters()
        ]
        
        if self.isTrain:
            self.loss_names = ['content', 'global', 'local']
            self.criterionMSE = torch.nn.MSELoss().to(self.device)
            self.optimizer_g = torch.optim.Adam(itertools.chain(*parameters), lr=opt.lr)
            self.optimizers.append(self.optimizer_g)
            
    def set_input(self, input_dict):
        self.c = input_dict['c'].to(self.device)  # content 音频
        self.s = input_dict['s'].to(self.device)  # style 音频
        self.image_paths = input_dict['name']
    
    # def encode_with_intermediate(self, x):
    #     self.audio_encoder = self.encodec_model.encode
    #     return self.audio_encoder(x)  # 直接使用音频编码器
    
    def get_key(self, feats, last_layer_idx):
        return networks.mean_variance_norm(feats[last_layer_idx])
    
    def forward(self):
        self.c_feats =  self.encodec_model.encode(self.c)
        self.s_feats = self.encodec_model.encode(self.s)
        if self.opt.skip_connection_3:
            c_adain_feat_3 = self.net_adaattn_3(self.c_feats[2], self.s_feats[2], self.get_key(self.c_feats, 2, self.opt.shallow_layer),
                                                   self.get_key(self.s_feats, 2, self.opt.shallow_layer), self.seed)
        else:
            c_adain_feat_3 = None

    
        cs = self.net_transformer(self.c_feats[3], self.s_feats[3], self.c_feats[4], self.s_feats[4],
                                  self.get_key(self.c_feats, 3, self.opt.shallow_layer),
                                  self.get_key(self.s_feats, 3, self.opt.shallow_layer),
                                  self.get_key(self.c_feats, 4, self.opt.shallow_layer),
                                  self.get_key(self.s_feats, 4, self.opt.shallow_layer), self.seed)
        
        self.cs = self.encodec_model.decode(cs)
    
    def compute_content_loss(self, stylized_feats):
        self.loss_content = self.criterionMSE(
            networks.mean_variance_norm(stylized_feats),
            networks.mean_variance_norm(self.c_feats)
        )
    
    def compute_style_loss(self, stylized_feats):
        s_feats_mean, s_feats_std = networks.calc_mean_std(self.s_feats)
        stylized_feats_mean, stylized_feats_std = networks.calc_mean_std(stylized_feats)
        self.loss_global = self.criterionMSE(stylized_feats_mean, s_feats_mean) + \
                           self.criterionMSE(stylized_feats_std, s_feats_std)
        
    def compute_losses(self):
        stylized_feats = self.encodec_model.encode(self.cs)
        self.compute_content_loss(stylized_feats)
        self.compute_style_loss(stylized_feats)
    
    def optimize_parameters(self):
        self.seed = int(torch.randint(10000000, (1,))[0])
        self.forward()
        self.optimizer_g.zero_grad()
        self.compute_losses()
        loss = self.loss_content + self.loss_global
        loss.backward()
        self.optimizer_g.step()
