import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from model import EncodecModel
#import modules as m
import networks
from itertools import product
import torchaudio
import model
import torchaudio
import torch
from models.base_model import BaseModel

class AdaAttNModel(BaseModel):
    def __init__(self,opt):
        BaseModel.__init__(self, opt)
        self.max_sample = 64 * 64  # 控制注意力计算的采样规模
        self.seed = 6666
        parameters = []
        # 适配 encodec 编码的音频输入
        target_bandwidths = [3, 6, 12, 24]
        self.encodec_model = EncodecModel.encodec_model_24khz()  # 直接实例化模型
        for bw in target_bandwidths:
            self.encodec_model.set_target_bandwidth(bw)  # 设置目标带宽

        if opt.skip_connection_3:
            adaattn_3 = networks.AdaAttN(in_planes=256, key_planes=256 + 128 + 64 if opt.shallow_layer else 256,
                                              max_sample=self.max_sample)
            self.net_adaattn_3 = networks.init_net(adaattn_3, opt.init_type, opt.init_gain, opt.gpu_ids)
            self.model_names.append('adaattn_3')
            parameters.append(self.net_adaattn_3.parameters())
        if opt.shallow_layer:
            channels = 512 + 256 + 128 + 64
        else:
            channels = 512
        transformer = networks.Transformer(in_planes=512, key_planes=channels, shallow_layer=opt.shallow_layer)
        self.net_transformer = networks.init_net(transformer, opt.init_type, opt.init_gain, opt.gpu_ids)
        decoder = networks.Decoder(opt.skip_connection_3)
        self.net_decoder = networks.init_net(decoder, opt.init_type, opt.init_gain, opt.gpu_ids)
        parameters.append(self.net_transformer.parameters())
        parameters.append(self.net_decoder.parameters())
        self.c = None
        self.cs = None
        self.s = None
        self.s_feats = None
        self.c_feats = None
        self.seed = 6666
        if self.isTrain:
            self.loss_names = ['content', 'global', 'local']
            self.criterionMSE = torch.nn.MSELoss().to(self.device)
            self.optimizer_g = torch.optim.Adam(itertools.chain(*parameters), lr=opt.lr)
            self.optimizers.append(self.optimizer_g)
            self.loss_global = torch.tensor(0., device=self.device)
            self.loss_local = torch.tensor(0., device=self.device)
            self.loss_content = torch.tensor(0., device=self.device)

    def set_input(self, input_dict):
        self.c = input_dict['c'].to(self.device)  # content 音频
        self.s = input_dict['s'].to(self.device)  # style 音频
        self.image_paths = input_dict['name']
    
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

        cs = self.net_transformer(self.c_feats, self.s_feats,self.get_key(self.c_feats, -1), self.get_key(self.s_feats, -1),self.seed)
        # cs = self.net_transformer(self.c_feats[3], self.s_feats[3], self.c_feats[4], self.s_feats[4],
        #                           self.get_key(self.c_feats, 3, self.opt.shallow_layer),
        #                           self.get_key(self.s_feats, 3, self.opt.shallow_layer),
        #                           self.get_key(self.c_feats, 4, self.opt.shallow_layer),
        #                           self.get_key(self.s_feats, 4, self.opt.shallow_layer), self.seed)
        
        self.cs = self.net_decoder(cs, c_adain_feat_3)
        #self.cs = self.encodec_model.decode(cs)
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

def test():
    from options.train_options import TrainOptions
    opt = TrainOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    model = AdaAttNModel(opt)
if __name__ == '__main__':
    test()
#def test():
#     bandwidths = [3, 6, 12, 24]  # 只测试 24kHz 的不同带宽
#     model = EncodecModel.encodec_model_24khz()  # 直接实例化模型

#     for bw in bandwidths:
#         model.set_target_bandwidth(bw)  # 设置目标带宽
#         audio_suffix = "24k"
#         wav, sr = torchaudio.load(f"test_{audio_suffix}.wav")

#         # 转换为单声道
#         if wav.shape[0] == 2:
#             wav = torch.mean(wav, dim=0, keepdim=True)  # [1, time]

#         # 截取 2s 音频
#         wav = wav[:, :model.sample_rate * 2]

#         # 添加 batch 维度
#         wav_in = wav.unsqueeze(0)  # [1, 1, time]

#         # 进行编码
#         encoded_frames = model.encode(wav_in)
#         wav_dec = model(wav_in)[0]  # 解码

#         # 打印编码信息
#         print(f"Bandwidth {bw} kbps, Encoded Frames: {len(encoded_frames)}")
#         for i, (codes, scale) in enumerate(encoded_frames):
#             print(f"  Frame {i}: Code shape: {codes.shape}, Scale: {scale}")

#         # 确保解码后的音频形状一致
#         assert wav.shape == wav_dec.shape, (wav.shape, wav_dec.shape)

# if __name__ == '__main__':
#     test()
# EncodedFrame = tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]

# class AdaAttNModel(nn.Module):
#     def __init__(self,
#                  encoder: m.SEANetEncoder,
#                  decoder: m.SEANetDecoder,
#                  quantizer: qt.ResidualVectorQuantizer,
#                  target_bandwidths: tp.List[float],
#                  sample_rate: int,
#                  channels: int,
#                  normalize: bool = False,
#                  segment: tp.Optional[float] = None,
#                  overlap: float = 0.01,
#                  name: str = 'unset',
#                  ):
        
#         super().__init__()
#         self.decoder = decoder
#         self.sample_rate = sample_rate
#         self.segment = segment
#         self.overlap = overlap
#         self.normalize = normalize
#         self.encoder = encoder
#         self.encoder = encoder
#         self.quantizer = quantizer
#         self.decoder = decoder
#         self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
#         self.bandwidth: tp.Optional[float] = None
#         self.frame_rate = math.ceil(self.sample_rate / np.prod(self.encoder.ratios)) #75
#         adaattn_3 = networks.AdaAttN(in_planes=256, key_planes=256 + 128 + 64, max_sample=256)
#         self.net_adaattn_3 = networks.init_net(adaattn_3, init_type='normal', init_gain=0.02, gpu_ids=())
#         transformer = networks.Transformer(
#             in_planes=512, key_planes=channels, shallow_layer=True)
#         decoder = networks.Decoder(opt.skip_connection_3)
#         self.net_decoder = networks.init_net(decoder, opt.init_type, opt.init_gain, opt.gpu_ids)
#         self.net_transformer = networks.init_net(transformer, opt.init_type, opt.init_gain, opt.gpu_ids)
#         self.c = None
#         self.cs = None
#         self.s = None
#         self.s_feats = None
#         self.c_feats = None
#         self.seed = 6666
#         if self.isTrain:
#             self.loss_names = ['content', 'global', 'local']
#             self.criterionMSE = torch.nn.MSELoss().to(self.device)
#             self.optimizer_g = torch.optim.Adam(itertools.chain(*parameters), lr=opt.lr)
#             self.optimizers.append(self.optimizer_g)
#             self.loss_global = torch.tensor(0., device=self.device)
#             self.loss_local = torch.tensor(0., device=self.device)
#             self.loss_content = torch.tensor(0., device=self.device)
#     @property
#     def segment_length(self) -> tp.Optional[int]:
#         if self.segment is None:
#             return None
#         return int(self.segment * self.sample_rate)
    
#     @property
#     def segment_stride(self) -> tp.Optional[int]:
#         segment_length = self.segment_length
#         if segment_length is None:
#             return None
#         return max(1, int((1 - self.overlap) * segment_length))

#     def encode(self, x: torch.Tensor) -> tp.List[EncodedFrame]:
#             """Given a tensor `x`, returns a list of frames containing
#             the discrete encoded codes for `x`, along with rescaling factors
#             for each segment, when `self.normalize` is True.

#             Each frames is a tuple `(codebook, scale)`, with `codebook` of
#             shape `[B, K, T]`, with `K` the number of codebooks.
#             """
#             assert x.dim() == 3
#             _, channels, length = x.shape
#             assert channels > 0 and channels <= 2
#             segment_length = self.segment_length 
#             if segment_length is None: #segment_length = 1*sample_rate
#                 segment_length = length
#                 stride = length
#             else:
#                 stride = self.segment_stride  # type: ignore
#                 assert stride is not None

#             encoded_frames: tp.List[EncodedFrame] = []
#             for offset in range(0, length, stride): # shift windows to choose data
#                 frame = x[:, :, offset: offset + segment_length]
#                 encoded_frames.append(self._encode_frame(frame))
#             return encoded_frames

#     def _encode_frame(self, x: torch.Tensor) -> EncodedFrame:
#         length = x.shape[-1] # tensor_cut or original
#         duration = length / self.sample_rate
#         assert self.segment is None or duration <= 1e-5 + self.segment

#         if self.normalize:
#             mono = x.mean(dim=1, keepdim=True)
#             volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
#             scale = 1e-8 + volume
#             x = x / scale
#             scale = scale.view(-1, 1)
#         else:
#             scale = None
#         emb = self.encoder(x) # [2,1,10000] -> [2,128,32]
#         #TODO: Encodec Trainer的training
#         if self.training:
#             return emb,scale
#         codes = self.quantizer.encode(emb, self.frame_rate, self.bandwidth)
#         codes = codes.transpose(0, 1)
#         # codes is [B, K, T], with T frames, K nb of codebooks.
#         return codes, scale
    
#     def decode(self, encoded_frames: tp.List[EncodedFrame]) -> torch.Tensor:
#         """Decode the given frames into a waveform.
#         Note that the output might be a bit bigger than the input. In that case,
#         any extra steps at the end can be trimmed.
#         """
#         segment_length = self.segment_length
#         if segment_length is None:
#             assert len(encoded_frames) == 1
#             return self._decode_frame(encoded_frames[0])

#         frames = [self._decode_frame(frame) for frame in encoded_frames]
#         return _linear_overlap_add(frames, self.segment_stride or 1)

#     def _decode_frame(self, encoded_frame: EncodedFrame) -> torch.Tensor:
#         codes, scale = encoded_frame
#         if self.training:
#             emb = codes
#         else:
#             codes = codes.transpose(0, 1)
#             emb = self.quantizer.decode(codes)
#         out = self.decoder(emb)
#         if scale is not None:
#             out = out * scale.view(-1, 1, 1)
#         return out

#     @staticmethod
#     def _get_model(target_bandwidths: tp.List[float],
#                    sample_rate: int = 24_000,
#                    channels: int = 1,
#                    causal: bool = True,
#                    model_norm: str = 'weight_norm',
#                    audio_normalize: bool = True,
#                    segment: tp.Optional[float] = None,
#                    name: str = 'unset',
#                    ratios=[8, 5, 4, 2]):
#         encoder = m.SEANetEncoder(channels=channels, norm=model_norm, causal=causal,ratios=ratios)
#         decoder = m.SEANetDecoder(channels=channels, norm=model_norm, causal=causal,ratios=ratios)
#         n_q = int(1000 * target_bandwidths[-1] // (math.ceil(sample_rate / encoder.hop_length) * 10)) # int(1000*24//(math.ceil(24000/320)*10))
#         quantizer = qt.ResidualVectorQuantizer(
#             dimension=encoder.dimension,
#             n_q=n_q,
#             bins=1024,
#         )
#         model = AdaAttNModel(
#             encoder,
#             decoder,
#             quantizer,
#             target_bandwidths,
#             sample_rate,
#             channels,
#             normalize=audio_normalize,
#             segment=segment,
#             name=name,
#         )
#         return model    
    
#     ##原始        
#     def set_input(self, input_dict):
#         self.c = input_dict['c'].to(self.device)  # content 音频
#         self.s = input_dict['s'].to(self.device)  # style 音频
#         self.image_paths = input_dict['name']
    
#     def get_key(self, feats, last_layer_idx):
#         return networks.mean_variance_norm(feats[last_layer_idx])
    
#     def forward(self):
#         self.c_feats = self.encode(self.c)
#         self.s_feats = self.encode(self.s)
#         channels = 512 + 256 + 128 + 64
#         cs = self.net_transformer(
#             self.c_feats, self.s_feats,
#             self.get_key(self.c_feats, -1), self.get_key(self.s_feats, -1),
#             self.seed
#         )
#         self.cs = self.decode(cs)

#     def compute_content_loss(self, stylized_feats):
#         self.loss_content = self.criterionMSE(
#             networks.mean_variance_norm(stylized_feats),
#             networks.mean_variance_norm(self.c_feats)
#         )
    
#     def compute_style_loss(self, stylized_feats):
#         s_feats_mean, s_feats_std = networks.calc_mean_std(self.s_feats)
#         stylized_feats_mean, stylized_feats_std = networks.calc_mean_std(stylized_feats)
#         self.loss_global = self.criterionMSE(stylized_feats_mean, s_feats_mean) + \
#                            self.criterionMSE(stylized_feats_std, s_feats_std)
        
#     def compute_losses(self):
#         stylized_feats = self.encode(self.cs)
#         self.compute_content_loss(stylized_feats)
#         self.compute_style_loss(stylized_feats)
    
#     def optimize_parameters(self):
#         self.seed = int(torch.randint(10000000, (1,))[0])
#         self.forward()
#         self.optimizer_g.zero_grad()
#         self.compute_losses()
#         loss = self.loss_content + self.loss_global
#         loss.backward()
#         self.optimizer_g.step()
