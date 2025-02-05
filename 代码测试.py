
import torch
import typing as tp
from torch import nn
import modules as m
import quantization as qt
import math
import numpy as np
from utils import _check_checksum, _linear_overlap_add, _get_checkpoint_url
EncodedFrame = tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]

class AudioEncoder(nn.Module):
    def __init__(self,
                 encoder: m.SEANetEncoder,
                 decoder: m.SEANetDecoder,
                 quantizer: qt.ResidualVectorQuantizer,
                 target_bandwidths: tp.List[float],
                 sample_rate: int,
                 channels: int,
                 normalize: bool = False,
                 segment: tp.Optional[float] = None,
                 overlap: float = 0.01,
                 name: str = 'unset'):
        super().__init__()
        self.decoder = decoder
        self.sample_rate = sample_rate
        self.segment = segment
        self.overlap = overlap
        self.normalize = normalize
        self.encoder = encoder
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
        self.bandwidth: tp.Optional[float] = None
        self.frame_rate = math.ceil(self.sample_rate / np.prod(self.encoder.ratios)) #75
    @property
    def segment_length(self) -> tp.Optional[int]:
        if self.segment is None:
            return None
        return int(self.segment * self.sample_rate)
    
    @property
    def segment_stride(self) -> tp.Optional[int]:
        segment_length = self.segment_length
        if segment_length is None:
            return None
        return max(1, int((1 - self.overlap) * segment_length))

    def encode(self, x: torch.Tensor) -> tp.List[EncodedFrame]:
            """Given a tensor `x`, returns a list of frames containing
            the discrete encoded codes for `x`, along with rescaling factors
            for each segment, when `self.normalize` is True.

            Each frames is a tuple `(codebook, scale)`, with `codebook` of
            shape `[B, K, T]`, with `K` the number of codebooks.
            """
            assert x.dim() == 3
            _, channels, length = x.shape
            assert channels > 0 and channels <= 2
            segment_length = self.segment_length 
            if segment_length is None: #segment_length = 1*sample_rate
                segment_length = length
                stride = length
            else:
                stride = self.segment_stride  # type: ignore
                assert stride is not None

            encoded_frames: tp.List[EncodedFrame] = []
            for offset in range(0, length, stride): # shift windows to choose data
                frame = x[:, :, offset: offset + segment_length]
                encoded_frames.append(self._encode_frame(frame))
            return encoded_frames

    def _encode_frame(self, x: torch.Tensor) -> EncodedFrame:
        length = x.shape[-1] # tensor_cut or original
        duration = length / self.sample_rate
        assert self.segment is None or duration <= 1e-5 + self.segment

        if self.normalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None

        emb = self.encoder(x) # [2,1,10000] -> [2,128,32]
        #TODO: Encodec Trainer的training
        if self.training:
            return emb,scale
        codes = self.quantizer.encode(emb, self.frame_rate, self.bandwidth)
        codes = codes.transpose(0, 1)
        # codes is [B, K, T], with T frames, K nb of codebooks.
        return codes, scale
    
    def decode(self, encoded_frames: tp.List[EncodedFrame]) -> torch.Tensor:
        """Decode the given frames into a waveform.
        Note that the output might be a bit bigger than the input. In that case,
        any extra steps at the end can be trimmed.
        """
        segment_length = self.segment_length
        if segment_length is None:
            assert len(encoded_frames) == 1
            return self._decode_frame(encoded_frames[0])

        frames = [self._decode_frame(frame) for frame in encoded_frames]
        return _linear_overlap_add(frames, self.segment_stride or 1)

    def _decode_frame(self, encoded_frame: EncodedFrame) -> torch.Tensor:
        codes, scale = encoded_frame
        if self.training:
            emb = codes
        else:
            codes = codes.transpose(0, 1)
            emb = self.quantizer.decode(codes)
        out = self.decoder(emb)
        if scale is not None:
            out = out * scale.view(-1, 1, 1)
        return out

    @staticmethod
    def _get_model(target_bandwidths: tp.List[float],
                   sample_rate: int = 24_000,
                   channels: int = 1,
                   causal: bool = True,
                   model_norm: str = 'weight_norm',
                   audio_normalize: bool = True,
                   segment: tp.Optional[float] = None,
                   name: str = 'unset',
                   ratios=[8, 5, 4, 2]):
        encoder = m.SEANetEncoder(channels=channels, norm=model_norm, causal=causal,ratios=ratios)
        decoder = m.SEANetDecoder(channels=channels, norm=model_norm, causal=causal,ratios=ratios)
        n_q = int(1000 * target_bandwidths[-1] // (math.ceil(sample_rate / encoder.hop_length) * 10)) # int(1000*24//(math.ceil(24000/320)*10))
        quantizer = qt.ResidualVectorQuantizer(
            dimension=encoder.dimension,
            n_q=n_q,
            bins=1024,
        )
        model = AudioEncoder(
            encoder,
            decoder,
            quantizer,
            target_bandwidths,
            sample_rate,
            channels,
            normalize=audio_normalize,
            segment=segment,
            name=name,
        )
        return model

x = torch.randn(2,1,10000)

target_bandwidths = [1.5, 3.0, 6.0]
print(x.size())
x_codec = AudioEncoder._get_model(target_bandwidths)
x_codec.eval()
with torch.no_grad():  # 关闭梯度计算
    encoded_frames = x_codec.encode(x)
    decoded_output = x_codec.decode(encoded_frames)
#print(x_encodec)
print("Encoded Frames:", len(encoded_frames))
print("decoded_output:", len(decoded_output))
for i, (codes, scale) in enumerate(encoded_frames):
    print(f"Frame {i}: Code shape: {codes.shape}, Scale: {scale}")
for i, output in enumerate(decoded_output):
    print(f"Frame {i}: Output shape: {output.shape}")