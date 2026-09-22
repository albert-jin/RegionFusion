"""Frozen C-RADIO spatial branch; external pretraining is an explicit extra cost."""
from pathlib import Path
import hashlib,json,sys
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
EXPECTED='23e0c117de49d4ce909150fe6658d470829e6639647c7a5b035ce82e0d5b763c'

class FrozenRadioSpatial(nn.Module):
    def __init__(self,device):
        super().__init__()
        runtime=ROOT/'ModuSeg_runtime';sys.path.insert(0,str(runtime))
        from C_RADIOv4_SO400M.hf_model import RADIOConfig,RADIOModel
        from safetensors.torch import load_file
        weights=ROOT/'shared/region_teacher/C_RADIOv4_SO400M.safetensors'
        h=hashlib.sha256()
        with weights.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        assert h.hexdigest()==EXPECTED,'Unexpected pretrained RADIO weights'
        raw=json.loads((runtime/'C_RADIOv4_SO400M/config.json').read_text())
        cfg=RADIOConfig(**{k:v for k,v in raw.items() if k not in ['auto_map','architectures','torch_dtype','transformers_version']})
        self.encoder=RADIOModel(cfg)
        self.encoder.load_state_dict(load_file(str(weights)),strict=True)
        self.encoder.to(device).eval().requires_grad_(False)
        self.register_buffer('image_mean',torch.tensor([123.675,116.28,103.53]).reshape(1,3,1,1),persistent=False)
        self.register_buffer('image_std',torch.tensor([58.395,57.12,57.375]).reshape(1,3,1,1),persistent=False)

    @torch.no_grad()
    def forward(self,normalized_image):
        self.encoder.eval()
        x=(normalized_image.float()*self.image_std+self.image_mean).div(255).clamp(0,1)
        h,w=x.shape[-2:]
        assert h%16==0 and w%16==0,'Fusion requires aligned CLIP/RADIO patch-16 grids'
        _,spatial=self.encoder(x)
        if spatial.ndim==3:
            assert spatial.shape[1]==(h//16)*(w//16)
            spatial=spatial.transpose(1,2).reshape(x.shape[0],-1,h//16,w//16)
        assert spatial.shape==(x.shape[0],1152,h//16,w//16)
        assert torch.isfinite(spatial).all()
        # Normalize per pixel without using any image tag or ground-truth label.
        spatial=F.layer_norm(spatial.permute(0,2,3,1).float(),(1152,)).permute(0,3,1,2).contiguous()
        return spatial
