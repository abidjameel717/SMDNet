# =============================================================================
# CELL 57 — MODEL SIZE, FLOPs AND INFERENCE TIME (Supplementary Table S2)
# SMD-Net V2 | Scientific Reports        No training, no dataset download.
# Uses the exact frozen definitions (code cell 49) and the Full_SSL checkpoint.
# =============================================================================
import ast, json, time, subprocess
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchaudio, cv2, shutil
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.flop_counter import FlopCounterMode
from google.colab import drive

drive.mount("/content/drive")
ROOT = Path("/content/drive/MyDrive/SMDNet_V2_ScientificReports")
assert torch.cuda.is_available(), "Switch to a T4 GPU runtime."
DEVICE = torch.device("cuda")

# --- frozen definitions from code cell 49 (same procedure as Cells 48-56)
nbk = sorted(ROOT.rglob("Video_forgery*.ipynb"),
             key=lambda p: (p.name != "Video_forgery (1).ipynb", -p.stat().st_size))[0]
code = ["".join(c["source"]) for c in json.loads(nbk.read_text())["cells"] if c["cell_type"] == "code"][49]
WINDOW_SEC, NUM_FRAMES, AUDIO_SR, AUDIO_SAMPLES = 0.8, 16, 16000, 12800
N_FFT, WIN_LENGTH, HOP_LENGTH, N_MELS, IMAGE_SIZE, RESIZE_SHORT = 640, 400, 160, 128, 224, 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=AUDIO_SR, n_fft=N_FFT, win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH, n_mels=N_MELS, f_min=0.0, f_max=8000.0, power=2.0, center=True)
ffmpeg_path = shutil.which("ffmpeg")
DECODER_STATS = {"opencv_success": 0, "ffmpeg_fallback": 0, "complete_failure": 0}
for n in ast.parse(code).body:
    if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
        m = ast.Module(body=[n], type_ignores=[]); ast.fix_missing_locations(m)
        exec(compile(m, f"cell49:{n.name}", "exec"), globals())

model = SMDNetV2(pretrained_visual=False)
ck = torch.load(ROOT / "04_Checkpoints/Full_SSL_Initialized_Fast/smdnet_v2_full_best.pt",
                map_location="cpu", weights_only=True)
model.load_state_dict(ck["model_state_dict"], strict=True)
model = model.to(DEVICE).eval()

# --- parameters
def count(mod): return sum(p.numel() for p in mod.parameters())
print(f"Total parameters: {count(model):,}")
for name, sub in model.named_children():
    print(f"  {name:<22s} {count(sub):>12,}")

# --- FLOPs (FlopCounterMode counts 2 x multiply-accumulate)
def flops(n_windows):
    v = torch.randn(n_windows, 16, 3, 224, 224, device=DEVICE)
    a = torch.randn(n_windows, 1, 128, 81, device=DEVICE)
    with torch.inference_mode(), FlopCounterMode(display=False) as fc:
        model(v, a)
    return fc.get_total_flops() / 1e9
g1, g5 = flops(1), flops(5)
print(f"\nGFLOPs per 0.8 s window: {g1:.2f}")
print(f"GFLOPs per video (5 windows): {g5:.2f}")

# --- GPU latency, model only, original AMP protocol (5-window batch)
v = torch.randn(5, 16, 3, 224, 224, device=DEVICE); a = torch.randn(5, 1, 128, 81, device=DEVICE)
times = []
with torch.inference_mode(), torch.amp.autocast(device_type="cuda"):
    for i in range(60):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        model(v, a)
        torch.cuda.synchronize()
        if i >= 10: times.append(time.perf_counter() - t0)       # 10 warm-up runs
ms = 1000 * np.array(times)
print(f"\nModel inference per video (5 windows, T4, AMP): {ms.mean():.1f} ± {ms.std():.1f} ms "
      f"(median {np.median(ms):.1f} ms, n = {len(ms)})")
print(f"Throughput: {1000 / ms.mean():.1f} videos/s  |  {5000 / ms.mean():.1f} windows/s")

# --- end-to-end per video including decoding, on the representative samples (if present)
samples = sorted((ROOT / "08_Representative_Dataset_Samples").rglob("*.mp4"))
if samples:
    e2e = []
    for p in samples:
        r = cv2.VideoCapture(str(p)); fps = r.get(cv2.CAP_PROP_FPS); n = r.get(cv2.CAP_PROP_FRAME_COUNT); r.release()
        dur = n / fps if fps else 0
        starts = np.array([0, .25, .5, .75, 1.0]) * max(0.0, dur - WINDOW_SEC)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        vb = torch.stack([decode_visual_segment(p, float(s), WINDOW_SEC, NUM_FRAMES) for s in starts]).to(DEVICE)
        ab = torch.stack([decode_audio_segment(p, float(s), WINDOW_SEC) for s in starts]).to(DEVICE)
        with torch.inference_mode(), torch.amp.autocast(device_type="cuda"):
            model(vb, ab)
        torch.cuda.synchronize(); e2e.append(time.perf_counter() - t0)
    print(f"End-to-end per video incl. decoding ({len(e2e)} sample videos): {1000 * np.mean(e2e):.0f} ms (mean)")
print("\nPaste this output back to fill Supplementary Table S2.")
