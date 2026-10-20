"""Frozen encoders: DINOv2 (mean-pool baseline) and DeltaTok (DINOv3 ViT-B + Kinetics tokenizer)."""
import sys, os, glob, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "deltatok"))
from transformers import AutoModel
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
def prep(imgs_uint8, size):
    """list of HxWx3 RGB uint8 -> (N,3,size,size) normalized, centre square crop then resize."""
    out = []
    for im in imgs_uint8:
        t = torch.from_numpy(im).permute(2, 0, 1).float() / 255; h, w = t.shape[1:]; s = min(h, w)
        t = t[:, (h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
        out.append(F.interpolate(t[None], size=(size, size), mode="bilinear", antialias=True, align_corners=False)[0])
    return (torch.stack(out) - MEAN) / STD
class DINOv2:
    def __init__(self, name="facebook/dinov2-base"): self.m = AutoModel.from_pretrained(name).eval()
    @torch.no_grad()
    def __call__(self, x):  # -> (N, D) CLS + mean patch concatenated? use CLS and patch-mean
        h = self.m(pixel_values=x).last_hidden_state
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], -1)
class DeltaTokEnc:
    """Returns per-frame DINOv3 patch-mean features and delta tokens z_t = enc(y_{t-1}, y_t) for t>=1."""
    def __init__(self):
        import types; _tb = types.ModuleType("training.base"); _tb.load_sd = lambda m, sd: m.load_state_dict(sd, strict=False)
        sys.modules.setdefault("training", types.ModuleType("training")); sys.modules["training.base"] = _tb  # avoid lightning/wandb import chain
        from models.dinov3 import DINOv3; from models.deltatok import DeltaTok
        ck = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Amazon-FAR--deltatok-kinetics/snapshots/*/pytorch_model.bin"))[0]
        self.m = DeltaTok(DINOv3()); sd = torch.load(ck, map_location="cpu")
        miss = self.m.load_state_dict(sd, strict=False)
        assert all(k.startswith("backbone.") for k in miss.missing_keys), miss.missing_keys[:5]
        self.m.eval()
    @torch.no_grad()
    def __call__(self, x):  # x: (T,3,S,S) normalized; returns feats (T,D), z (T-1,D), recon_cos (T-1,)
        bb = self.m.backbone.backbone; npre = self.m.backbone.num_prefix_tokens
        y = bb(pixel_values=x).last_hidden_state[:, npre:]
        rope = self.m.rope_embeddings(x[:1].float())
        z = self.m.encode(y[:-1], y[1:], rope)
        y_hat = self.m.decode(z, y[:-1], rope)
        rc = F.cosine_similarity(y_hat, y[1:], -1).mean(1); cc = F.cosine_similarity(y[:-1], y[1:], -1).mean(1)
        return y.mean(1), z[:, 0], rc, cc
