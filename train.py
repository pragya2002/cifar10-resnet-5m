"""CIFAR-10 ResNet challenge, training script v2.

Modes: student | teacher | kd.  Archs: wrn (pre-activation Wide ResNet, optional SE) | shake (Shake-Shake 26 2xWd) | resnet (4-stage pre-act ResNet-18 style).
Augment: plain (crop+flip only) | ta (TrivialAugment) | robust (TrivialAugment or AugMix + color jitter + grayscale + sharpness)
         | scale (TrivialAugment + zoom/RandomResizedCrop + random down-up resampling + blur + JPEG; v2.2).
v2.1 fixes: SAM second pass does not update BatchNorm running stats; best raw / best EMA checkpoints saved;
explicit TTA shift with matching saved val predictions; cache vs CSV alignment checks.
KD supports an ensemble of teachers (average of softened probabilities).
Every run writes to OUT_DIR/RUN_NAME: last.pt (resumable), final_ema.pt, log.csv, config.json,
summary.json, submission.csv, test_probs.npy, val_probs.npy, best_raw.pt, best_ema.pt.
"""
import argparse, glob, json, math, os, random, sys, time
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torchvision.transforms import v2

p = argparse.ArgumentParser()
p.add_argument('--mode', default='student', choices=['student', 'teacher', 'kd'])
p.add_argument('--run_name', required=True)
p.add_argument('--out_dir', default='/kaggle/working')
p.add_argument('--data_root', default=None)
p.add_argument('--cache', default=None, help='path of data_cache.npz (created if missing)')
p.add_argument('--seed', type=int, default=42)
p.add_argument('--split_seed', type=int, default=42)
p.add_argument('--val_size', type=int, default=5000)
p.add_argument('--full_train', action='store_true')
p.add_argument('--arch', default='wrn', choices=['wrn', 'shake', 'resnet', 'pyramid'])
p.add_argument('--depth', type=int, default=22)
p.add_argument('--widths', default='72,144,264', help='wrn: 3 stage widths; shake: base width d (2xd); resnet: 4 stage widths; pyramid: alpha (widening)')
p.add_argument('--blocks', default='2,2,2,2', help='resnet: residual blocks per stage')
p.add_argument('--dropout', type=float, default=0.0)
p.add_argument('--shakedrop', type=float, default=0.5, help='pyramid: ShakeDrop p of the last block (linear decay rule)')
p.add_argument('--se', action='store_true', help='squeeze-and-excitation in wrn blocks')
p.add_argument('--epochs', type=int, default=200)
p.add_argument('--bs', type=int, default=128)
p.add_argument('--lr', type=float, default=0.1)
p.add_argument('--wd', type=float, default=5e-4)
p.add_argument('--warmup', type=int, default=5)
p.add_argument('--ls', type=float, default=0.1)
p.add_argument('--sam', action='store_true')
p.add_argument('--rho', type=float, default=0.05)
p.add_argument('--aug', default='ta', choices=['plain', 'ta', 'robust', 'scale'])
p.add_argument('--erase', type=float, default=0.5)
p.add_argument('--mix_prob', type=float, default=0.5)
p.add_argument('--mix_off', type=float, default=0.1)
p.add_argument('--ema', type=float, default=0.999)
p.add_argument('--teachers', default='', help='comma separated teacher final_ema.pt paths (mode kd)')
p.add_argument('--kd_alpha', type=float, default=0.8)
p.add_argument('--kd_T', type=float, default=4.0)
p.add_argument('--kd_w', type=float, default=1.0, help='multiplier on the KD term (logit std paper uses 9)')
p.add_argument('--kd_std', action='store_true', help='logit standardization (Sun et al. CVPR 2024): z-score logits before /T')
p.add_argument('--eval_every', type=int, default=5)
p.add_argument('--tta_shift', type=int, default=-1, help='-1: choose on val (or 1 when full_train); else fixed 0/1/2')
p.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 2))
p.add_argument('--no_amp', action='store_true')
p.add_argument('--param_limit', type=int, default=5_000_000)
p.add_argument('--unlabeled_test', action='store_true', help='ONLY IF WRITTEN PERMISSION: add the unlabeled test images to KD training (teacher targets only, no labels)')
p.add_argument('--unl_repeat', type=int, default=5, help='how many times the unlabeled test images are repeated per epoch')
p.add_argument('--init', default='', help='start from this checkpoint (final_ema.pt) instead of random init')
p.add_argument('--smoke', action='store_true', help='tiny quick run to test the pipeline')
args = p.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
AMP = DEV == 'cuda' and not args.no_amp
torch.backends.cudnn.benchmark = True
RUN = os.path.join(args.out_dir, args.run_name); os.makedirs(RUN, exist_ok=True)
json.dump(vars(args), open(os.path.join(RUN, 'config.json'), 'w'), indent=2)
log = lambda *a: print(*a, flush=True)
log('device', DEV, torch.cuda.get_device_name(0) if DEV == 'cuda' else '', '| args', vars(args))

# ------------------------------------------------------------------ data
def find_root():
    if args.data_root: return args.data_root
    for base in ['/kaggle/input', '/content/data', './data']:
        hits = glob.glob(os.path.join(base, '**', 'train_labels.csv'), recursive=True)
        if hits: return os.path.dirname(hits[0])
    raise FileNotFoundError('train_labels.csv not found; pass --data_root')
ROOT = find_root()
train_df = pd.read_csv(os.path.join(ROOT, 'train_labels.csv'))
test_df = pd.read_csv(os.path.join(ROOT, 'test.csv'))
sample_sub = pd.read_csv(os.path.join(ROOT, 'sample_submission.csv'))
cache = args.cache or os.path.join(args.out_dir, 'data_cache.npz')
if os.path.exists(cache):
    z = np.load(cache); X_all, y_all, X_test = z['X'], z['y'], z['Xt']
else:
    idx = {f: os.path.join(d, f) for d, _, fs in os.walk(ROOT) for f in fs if f.endswith('.png')}
    load = lambda fs: np.stack([np.asarray(Image.open(idx[os.path.basename(f)]).convert('RGB')) for f in fs])
    X_all, y_all, X_test = load(train_df['filename']), train_df['label'].values.astype(np.int64), load(test_df['filename'])
    np.savez(cache, X=X_all, y=y_all, Xt=X_test)
# ---- alignment checks: cache must match the CSVs (order, labels) and, where PNGs exist, the image files
assert len(X_all) == len(train_df) and len(X_test) == len(test_df) == len(sample_sub), 'cache/CSV length mismatch'
assert (y_all == train_df['label'].values).all(), 'cached labels differ from train_labels.csv'
assert set(test_df['id']) == set(sample_sub['id']) and test_df['id'].is_unique, 'test ids differ from sample_submission'
_png = {f: os.path.join(d, f) for d, _, fs in os.walk(ROOT) for f in fs if f.endswith('.png')}
if _png:
    _r = np.random.RandomState(0)
    for i in _r.choice(len(train_df), 50, replace=False):
        assert (np.asarray(Image.open(_png[os.path.basename(train_df['filename'][i])]).convert('RGB')) == X_all[i]).all(), f'train cache row {i} != png'
    for i in _r.choice(len(test_df), 50, replace=False):
        assert (np.asarray(Image.open(_png[os.path.basename(test_df['filename'][i])]).convert('RGB')) == X_test[i]).all(), f'test cache row {i} != png'
    log('alignment check: 100 random cached images match their PNGs')
else:
    log('alignment check: PNGs not present here, checked lengths/labels/ids only')
MEAN = X_all.reshape(-1, 3).mean(0, dtype=np.float64) / 255.0
STD = X_all.reshape(-1, 3).std(0, dtype=np.float64) / 255.0
rng = np.random.RandomState(args.split_seed)   # identical split to the v1 notebook (seed 42)
val_idx = np.concatenate([rng.permutation(np.where(y_all == c)[0])[:args.val_size // 10] for c in range(10)])
tr_idx = np.arange(len(y_all)) if args.full_train else np.setdiff1d(np.arange(len(y_all)), val_idx)
if args.full_train: val_idx = np.array([], dtype=int)
if args.smoke:
    tr_idx = tr_idx[np.random.RandomState(0).permutation(len(tr_idx))[:2048]]
    val_idx = val_idx[::10]; args.epochs = 2; args.eval_every = 1
log('train', len(tr_idx), 'val', len(val_idx), 'test', len(X_test))

class ArrDS(Dataset):
    def __init__(s, X, y, tf): s.X = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous(); s.y = None if y is None else torch.from_numpy(y); s.tf = tf
    def __len__(s): return len(s.X)
    def __getitem__(s, i):
        x = s.tf(s.X[i]); return (x, s.y[i]) if s.y is not None else x

norm = [v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN.tolist(), STD.tolist())]
class RandomDownUp:
    # shrink to a random size then back to 32: simulates different resampling / source resolution
    def __init__(s, p=0.25, lo=20, hi=30): s.p, s.lo, s.hi = p, lo, hi
    def __call__(s, x):
        if random.random() >= s.p: return x
        k = random.randint(s.lo, s.hi)
        return v2.functional.resize(v2.functional.resize(x, [k, k], antialias=True), [32, 32], antialias=True)
base = [v2.RandomCrop(32, padding=4, padding_mode='reflect'), v2.RandomHorizontalFlip()]
if args.aug == 'scale':   # half the time a zoom (scale 0.4..1, aspect 3/4..4/3) instead of the pad-and-crop shift
    base = [v2.RandomChoice([v2.RandomCrop(32, padding=4, padding_mode='reflect'),
                             v2.RandomResizedCrop(32, scale=(0.4, 1.0), ratio=(0.75, 1.333), antialias=True)]),
            v2.RandomHorizontalFlip()]
if args.aug == 'plain':
    aug = []
elif args.aug == 'ta':
    aug = [v2.TrivialAugmentWide()]
elif args.aug == 'scale':
    aug = [v2.TrivialAugmentWide(), RandomDownUp(p=0.25),
           v2.RandomApply([v2.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.15),
           v2.RandomApply([v2.JPEG(quality=(40, 95))], p=0.3)]
else:
    aug = [v2.RandomChoice([v2.TrivialAugmentWide(), v2.AugMix(severity=3, mixture_width=3)]),
           v2.RandomApply([v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)], p=0.5),
           v2.RandomGrayscale(p=0.05), v2.RandomAdjustSharpness(2.0, p=0.3)]
erase = [v2.RandomErasing(p=args.erase, scale=(0.02, 0.25), value=0)] if args.erase > 0 else []
train_tf = v2.Compose(base + aug + norm + erase)
test_tf = v2.Compose(norm)
pin = DEV == 'cuda'
X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
if args.unlabeled_test:   # unlabeled test images: label -1, so only the teacher (KD) term trains on them
    assert args.mode == 'kd', '--unlabeled_test needs teachers'
    X_tr = np.concatenate([X_tr] + [X_test] * args.unl_repeat); y_tr = np.concatenate([y_tr, -np.ones(len(X_test) * args.unl_repeat, dtype=y_tr.dtype)])
    log('unlabeled test images added:', len(X_test) * args.unl_repeat, '| train set now', len(X_tr))
train_dl = DataLoader(ArrDS(X_tr, y_tr, train_tf), batch_size=args.bs, shuffle=True, drop_last=True,
                      num_workers=args.workers, pin_memory=pin, persistent_workers=args.workers > 0)
val_dl = DataLoader(ArrDS(X_all[val_idx], y_all[val_idx], test_tf), batch_size=512, num_workers=2, pin_memory=pin) if len(val_idx) else None

# ------------------------------------------------------------------ models
class SE(nn.Module):
    def __init__(s, c, r=16):
        super().__init__(); s.f1 = nn.Conv2d(c, max(8, c // r), 1); s.f2 = nn.Conv2d(max(8, c // r), c, 1)
    def forward(s, x): return x * torch.sigmoid(s.f2(F.relu(s.f1(F.adaptive_avg_pool2d(x, 1)))))

class PreActBlock(nn.Module):
    def __init__(s, cin, cout, stride, dropout, se):
        super().__init__()
        s.bn1 = nn.BatchNorm2d(cin); s.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        s.bn2 = nn.BatchNorm2d(cout); s.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        s.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        s.se = SE(cout) if se else nn.Identity()
        s.shortcut = None if (stride == 1 and cin == cout) else nn.Conv2d(cin, cout, 1, stride, bias=False)
    def forward(s, x):
        o = F.relu(s.bn1(x)); sc = x if s.shortcut is None else s.shortcut(o)
        o = s.conv2(s.drop(F.relu(s.bn2(s.conv1(o)))))
        return s.se(o) + sc

class PreActWRN(nn.Module):
    def __init__(s, depth=22, widths=(72, 144, 264), dropout=0.0, se=False, num_classes=10, stem=16):
        super().__init__(); n = (depth - 4) // 6; assert (depth - 4) % 6 == 0
        s.stem = nn.Conv2d(3, stem, 3, 1, 1, bias=False); L, cin = [], stem
        for i, w in enumerate(widths):
            for j in range(n): L.append(PreActBlock(cin, w, 2 if (i > 0 and j == 0) else 1, dropout, se)); cin = w
        s.blocks = nn.Sequential(*L); s.bn = nn.BatchNorm2d(cin); s.fc = nn.Linear(cin, num_classes)
        _init(s)
    def forward(s, x):
        x = F.relu(s.bn(s.blocks(s.stem(x)))); return s.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))

class PreActResNet4(nn.Module):
    # ResNet-18 style: 3x3 stem (stride 1, no max-pool), 4 stages, pre-activation basic blocks
    def __init__(s, widths=(64, 128, 256, 240), blocks=(2, 2, 2, 2), num_classes=10, stem=64):
        super().__init__(); s.stem = nn.Conv2d(3, stem, 3, 1, 1, bias=False); L, cin = [], stem
        for i, (w, n) in enumerate(zip(widths, blocks)):
            for j in range(n): L.append(PreActBlock(cin, w, 2 if (i > 0 and j == 0) else 1, 0.0, False)); cin = w
        s.blocks = nn.Sequential(*L); s.bn = nn.BatchNorm2d(cin); s.fc = nn.Linear(cin, num_classes)
        _init(s)
    def forward(s, x):
        x = F.relu(s.bn(s.blocks(s.stem(x)))); return s.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))

class _Shake(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, training):
        alpha = a.new_empty(a.size(0), 1, 1, 1).uniform_() if training else a.new_full((a.size(0), 1, 1, 1), 0.5)
        return alpha * a + (1 - alpha) * b
    @staticmethod
    def backward(ctx, g):
        beta = g.new_empty(g.size(0), 1, 1, 1).uniform_()
        return beta * g, (1 - beta) * g, None

class ShakeShortcut(nn.Module):
    def __init__(s, cin, cout, stride):
        super().__init__(); s.stride = stride
        s.c1 = nn.Conv2d(cin, cout // 2, 1, bias=False); s.c2 = nn.Conv2d(cin, cout - cout // 2, 1, bias=False); s.bn = nn.BatchNorm2d(cout)
    def forward(s, x):
        h = F.relu(x)
        h1 = F.avg_pool2d(h, 1, s.stride)
        h2 = F.avg_pool2d(F.pad(h, (-1, 1, -1, 1)), 1, s.stride)
        return s.bn(torch.cat([s.c1(h1), s.c2(h2)], 1))

class ShakeBlock(nn.Module):
    def __init__(s, cin, cout, stride):
        super().__init__()
        br = lambda: nn.Sequential(nn.ReLU(), nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout),
                                   nn.ReLU(), nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))
        s.b1, s.b2 = br(), br()
        s.sc = None if (cin == cout and stride == 1) else ShakeShortcut(cin, cout, stride)
    def forward(s, x):
        h = _Shake.apply(s.b1(x), s.b2(x), s.training)
        return h + (x if s.sc is None else s.sc(x))

class ShakeResNet(nn.Module):
    def __init__(s, depth=26, w=40, num_classes=10):
        super().__init__(); n = (depth - 2) // 6
        s.stem = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16)); L, cin = [], 16
        for i, c in enumerate([w, 2 * w, 4 * w]):
            for j in range(n): L.append(ShakeBlock(cin, c, 2 if (i > 0 and j == 0) else 1)); cin = c
        s.blocks = nn.Sequential(*L); s.fc = nn.Linear(cin, num_classes)
        _init(s)
    def forward(s, x):
        x = F.relu(s.blocks(s.stem(x))); return s.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))

class _ShakeDrop(torch.autograd.Function):
    # Yamada et al. 2019: out = (b + alpha - b*alpha) * x, b ~ Bernoulli(1-p), alpha ~ U[-1,1]; backward uses beta ~ U[0,1]
    @staticmethod
    def forward(ctx, x, training, p):
        if not training: return x * (1 - p)
        gate = (torch.rand(x.size(0), 1, 1, 1, device=x.device) >= p).to(x.dtype)
        alpha = torch.empty(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype).uniform_(-1, 1)
        ctx.save_for_backward(gate)
        return x * (gate + (1 - gate) * alpha)
    @staticmethod
    def backward(ctx, g):
        gate, = ctx.saved_tensors
        beta = torch.rand(g.size(0), 1, 1, 1, device=g.device, dtype=g.dtype)
        return g * (gate + (1 - gate) * beta), None, None

class PyrBottleneck(nn.Module):
    def __init__(s, cin, planes, stride, p):
        super().__init__(); s.stride, s.p = stride, p
        s.f = nn.Sequential(nn.BatchNorm2d(cin), nn.Conv2d(cin, planes, 1, bias=False), nn.BatchNorm2d(planes), nn.ReLU(inplace=True),
                            nn.Conv2d(planes, planes, 3, stride, 1, bias=False), nn.BatchNorm2d(planes), nn.ReLU(inplace=True),
                            nn.Conv2d(planes, planes * 4, 1, bias=False), nn.BatchNorm2d(planes * 4))
    def forward(s, x):
        h = _ShakeDrop.apply(s.f(x), s.training, s.p)
        sc = x if s.stride == 1 else F.avg_pool2d(x, 2)
        if sc.size(1) < h.size(1): sc = F.pad(sc, (0, 0, 0, 0, 0, h.size(1) - sc.size(1)))
        return h + sc

class PyramidNet(nn.Module):
    # bottleneck PyramidNet with ShakeDrop (Han et al. 2017, Yamada et al. 2019). 272 / alpha 200 = 26.2M params
    def __init__(s, depth=272, alpha=200, pL=0.5, num_classes=10):
        super().__init__(); n = (depth - 2) // 9; add = alpha / (3 * n); planes, cin, L, i = 16.0, 16, [], 0
        s.stem = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16))
        for st in range(3):
            for j in range(n):
                planes += add; i += 1; p = int(round(planes))
                L.append(PyrBottleneck(cin, p, 2 if (st > 0 and j == 0) else 1, pL * i / (3 * n))); cin = 4 * p
        s.blocks = nn.Sequential(*L); s.bn = nn.BatchNorm2d(cin); s.fc = nn.Linear(cin, num_classes)
        _init(s)
        for b in L: nn.init.zeros_(b.f[-1].weight)   # zero init last BN of each residual branch
    def forward(s, x):
        x = F.relu(s.bn(s.blocks(s.stem(x)))); return s.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))

def _init(m):
    for mod in m.modules():
        if isinstance(mod, nn.Conv2d): nn.init.kaiming_normal_(mod.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(mod, nn.BatchNorm2d): nn.init.ones_(mod.weight); nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.Linear): nn.init.zeros_(mod.bias)

def build(spec):
    if spec['arch'] == 'pyramid': return PyramidNet(spec['depth'], spec['alpha'], spec.get('pL', 0.5))
    if spec['arch'] == 'shake': return ShakeResNet(spec['depth'], spec['w'])
    if spec['arch'] == 'resnet': return PreActResNet4(tuple(spec['widths']), tuple(spec['blocks']))
    return PreActWRN(spec['depth'], tuple(spec['widths']), spec.get('dropout', 0.0), spec.get('se', False))

def spec_from_args():
    if args.arch == 'pyramid': return dict(arch='pyramid', depth=args.depth, alpha=int(args.widths), pL=args.shakedrop)
    if args.arch == 'shake': return dict(arch='shake', depth=args.depth, w=int(args.widths))
    if args.arch == 'resnet': return dict(arch='resnet', widths=[int(v) for v in args.widths.split(',')], blocks=[int(v) for v in args.blocks.split(',')])
    return dict(arch='wrn', depth=args.depth, widths=[int(v) for v in args.widths.split(',')], dropout=args.dropout, se=args.se)

def load_teacher(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if 'arch_spec' in ck: spec = ck['arch_spec']
    else:  # v1 notebook checkpoint
        c = ck['cfg']; spec = dict(arch='wrn', depth=c['T_DEPTH'], widths=list(c['T_WIDTHS']), dropout=c['T_DROPOUT'], se=False)
    m = build(spec); m.load_state_dict(ck['ema'] if 'ema' in ck else ck['model'])
    m = m.to(DEV).to(memory_format=torch.channels_last).eval()
    for q in m.parameters(): q.requires_grad_(False)
    return m

SPEC = spec_from_args()
model = build(SPEC)
n_params = sum(q.numel() for q in model.parameters() if q.requires_grad)
log(f'{args.mode} {SPEC} trainable params: {n_params:,}')
if args.mode != 'teacher':
    assert n_params <= args.param_limit, f'over budget: {n_params:,}'
if args.init:
    _ck = torch.load(args.init, map_location='cpu', weights_only=False); model.load_state_dict(_ck['ema'] if 'ema' in _ck else _ck['model'])
    log('initialised from', args.init)
model = model.to(DEV).to(memory_format=torch.channels_last)
teachers = [load_teacher(t) for t in args.teachers.split(',') if t] if args.mode == 'kd' else []
if args.mode == 'kd': assert teachers, 'mode kd needs --teachers'
log('teachers:', len(teachers))

# ------------------------------------------------------------------ training utils
def soft_targets(y, eps):
    t = torch.full((y.size(0), 10), eps / 10, device=y.device).scatter_(1, y.clamp(min=0)[:, None], 1 - eps + eps / 10)
    t[y < 0] = 0   # unlabeled rows carry no label target
    return t

def mix_batch(x, t):
    perm = torch.randperm(x.size(0), device=x.device)
    if random.random() < 0.5:
        lam = np.random.beta(1.0, 1.0); H, W = x.shape[2:]
        rh, rw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam)); cy, cx = np.random.randint(H), np.random.randint(W)
        y1, y2, x1, x2 = max(cy - rh // 2, 0), min(cy + rh // 2, H), max(cx - rw // 2, 0), min(cx + rw // 2, W)
        x = x.clone(); x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]; lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    else:
        lam = np.random.beta(0.2, 0.2); x = lam * x + (1 - lam) * x[perm]
    return x, lam * t + (1 - lam) * t[perm]

dec = [q for q in model.parameters() if q.ndim > 1]; nodec = [q for q in model.parameters() if q.ndim <= 1]
opt = torch.optim.SGD([{'params': dec, 'weight_decay': args.wd}, {'params': nodec, 'weight_decay': 0.0}],
                      lr=args.lr, momentum=0.9, nesterov=True)
spe = len(train_dl); total = args.epochs * spe; warm = args.warmup * spe
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
scaler = torch.amp.GradScaler('cuda', enabled=AMP)
ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True)

def zs(z):   # identity unless --kd_std
    return (z - z.mean(1, keepdim=True)) / (z.std(1, keepdim=True) + 1e-7) if args.kd_std else z

def train_step(x, y, mixing):
    t = soft_targets(y, args.ls)
    if mixing and random.random() < args.mix_prob: x, t = mix_batch(x, t)
    tp = None
    if teachers:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=AMP):
            tp = torch.stack([F.softmax(zs(tm(x).float()) / args.kd_T, 1) for tm in teachers]).mean(0)
    def loss_fn():
        with torch.autocast('cuda', dtype=torch.float16, enabled=AMP):
            out = model(x)
        out = out.float(); l = torch.sum(-t * F.log_softmax(out, 1), 1).mean()
        if tp is not None:
            kd = F.kl_div(F.log_softmax(zs(out) / args.kd_T, 1), tp, reduction='batchmean') * args.kd_T ** 2 * args.kd_w
            l = args.kd_alpha * kd + (1 - args.kd_alpha) * l
        return l, out
    loss, out = loss_fn(); scaler.scale(loss).backward()
    if args.sam:
        sc = scaler.get_scale() if AMP else 1.0
        ps = [q for q in model.parameters() if q.grad is not None]
        gn = torch.norm(torch.stack([(q.grad.detach() / sc).norm(2) for q in ps]), 2)
        if torch.isfinite(gn):
            eps = []
            with torch.no_grad():
                for q in ps: e = (q.grad / sc) * (args.rho / (gn + 1e-12)); q.add_(e); eps.append(e)
            opt.zero_grad(set_to_none=True)
            bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
            moms = [m.momentum for m in bns]
            for m in bns: m.momentum = 0.0   # perturbed pass: batch statistics only, running stats untouched
            l2, _ = loss_fn(); scaler.scale(l2).backward()
            for m, mo in zip(bns, moms): m.momentum = mo
            with torch.no_grad():
                for q, e in zip(ps, eps): q.sub_(e)
    scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); ema.update_parameters(model)
    return loss.item(), (out.argmax(1) == t.argmax(1)).float().mean().item()

@torch.no_grad()
def evaluate(net):
    net.eval(); c = n = 0
    for x, y in val_dl:
        x = x.to(DEV, non_blocking=True).to(memory_format=torch.channels_last); y = y.to(DEV)
        with torch.autocast('cuda', dtype=torch.float16, enabled=AMP): c += (net(x).argmax(1) == y).sum().item()
        n += y.numel()
    return c / n

# ------------------------------------------------------------------ train
last, start, best, rows = os.path.join(RUN, 'last.pt'), 0, -1.0, []
if os.path.exists(last) and not args.smoke:
    ck = torch.load(last, map_location=DEV, weights_only=False)
    model.load_state_dict(ck['model']); ema.module.load_state_dict(ck['ema']); ema.n_averaged.fill_(ck['n_averaged'])
    opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched']); scaler.load_state_dict(ck['scaler'])
    start, best, rows = ck['epoch'] + 1, ck['best'], ck['log']; log('resumed at epoch', start)
mix_off = int(args.epochs * (1 - args.mix_off))
for ep in range(start, args.epochs):
    model.train(); t0 = time.time(); tl = ta = 0.0
    for x, y in train_dl:
        x = x.to(DEV, non_blocking=True).to(memory_format=torch.channels_last); y = y.to(DEV, non_blocking=True)
        l, a = train_step(x, y, ep < mix_off); tl += l; ta += a
    row = dict(epoch=ep, loss=round(tl / spe, 4), train_acc=round(ta / spe, 4), lr=round(sched.get_last_lr()[0], 5), sec=round(time.time() - t0, 1))
    if val_dl is not None and ((ep + 1) % args.eval_every == 0 or ep == args.epochs - 1):
        row['val'] = evaluate(model); row['val_ema'] = evaluate(ema.module)
        if row['val_ema'] > best: best = row['val_ema']
        for key, net_, fn in [('val', model, 'best_raw.pt'), ('val_ema', ema.module, 'best_ema.pt')]:
            if row[key] >= max([r.get(key, -1) for r in rows] + [-1]):
                torch.save({'ema': net_.state_dict(), 'arch_spec': SPEC, 'cfg': vars(args), 'params': n_params, 'epoch': ep, key: row[key]}, os.path.join(RUN, fn))
    rows.append(row); log(row)
    torch.save({'model': model.state_dict(), 'ema': ema.module.state_dict(), 'n_averaged': ema.n_averaged, 'opt': opt.state_dict(),
                'sched': sched.state_dict(), 'scaler': scaler.state_dict(), 'epoch': ep, 'best': best, 'log': rows}, last + '.tmp')
    os.replace(last + '.tmp', last)
    pd.DataFrame(rows).to_csv(os.path.join(RUN, 'log.csv'), index=False)
final = os.path.join(RUN, 'final_ema.pt')
torch.save({'ema': ema.module.state_dict(), 'arch_spec': SPEC, 'cfg': vars(args), 'params': n_params}, final)

# ------------------------------------------------------------------ predict
@torch.no_grad()
def predict(net, X, shift):
    net.eval(); x = torch.from_numpy(X).permute(0, 3, 1, 2).float().div(255)
    x = (x - torch.tensor(MEAN, dtype=torch.float32)[:, None, None]) / torch.tensor(STD, dtype=torch.float32)[:, None, None]
    out = torch.zeros(len(x), 10)
    for i in range(0, len(x), 500):
        xb = x[i:i + 500].to(DEV); views = [xb, xb.flip(3)]
        if shift:
            pad = F.pad(xb, (shift,) * 4, mode='reflect')
            for dy, dx in [(0, shift), (2 * shift, shift), (shift, 0), (shift, 2 * shift)]:
                v = pad[:, :, dy:dy + 32, dx:dx + 32]; views += [v, v.flip(3)]
        acc = 0
        for v in views:
            with torch.autocast('cuda', dtype=torch.float16, enabled=AMP):
                acc = acc + F.softmax(net(v.contiguous(memory_format=torch.channels_last)).float(), 1)
        out[i:i + 500] = (acc / len(views)).cpu()
    return out.numpy()

net = ema.module
summary = dict(run=args.run_name, mode=args.mode, spec=SPEC, params=n_params, epochs=args.epochs, full_train=args.full_train)
shift = args.tta_shift if args.tta_shift >= 0 else 1
if len(val_idx):
    res, pvs = {}, {}
    for s in [0, 1, 2]:
        pvs[s] = predict(net, X_all[val_idx], s); res[s] = float((pvs[s].argmax(1) == y_all[val_idx]).mean())
    if args.tta_shift < 0: shift = max(res, key=res.get)
    np.save(os.path.join(RUN, 'val_probs.npy'), pvs[shift])   # matches the shift used for the submission
    summary.update(val_ema_plain=rows[-1].get('val_ema'), val_raw_plain=rows[-1].get('val'), val_tta=res)
    log('val TTA by shift:', res)
summary['tta_shift_used'] = shift; log('TTA shift used for test:', shift)
tp = predict(net, X_test, shift); np.save(os.path.join(RUN, 'test_probs.npy'), tp)
sub = sample_sub[['id']].merge(pd.DataFrame({'id': test_df['id'].values, 'label': tp.argmax(1)}), on='id', how='left')
assert sub['label'].notna().all() and len(sub) == len(sample_sub)
sub['label'] = sub['label'].astype(int); sub.to_csv(os.path.join(RUN, 'submission.csv'), index=False)
if len(val_idx):   # robustness on OUR validation images (same perturbations as diagnostics_colab), no TTA
    TFn = v2.functional; Xv = torch.from_numpy(X_all[val_idx]).permute(0, 3, 1, 2).contiguous()
    pert = {'zoom in 24->32': lambda x: TFn.resize(TFn.center_crop(x, [24, 24]), [32, 32], antialias=True),
            'down-up 24': lambda x: TFn.resize(TFn.resize(x, [24, 24], antialias=True), [32, 32], antialias=True),
            'blur s=0.8': lambda x: TFn.gaussian_blur(x, [3, 3], [0.8, 0.8]),
            'jpeg q75': lambda x: TFn.jpeg(x, 75), 'grayscale': lambda x: TFn.rgb_to_grayscale(x, num_output_channels=3)}
    rob = {}
    for k, f in pert.items():
        try: rob[k] = float((predict(net, f(Xv).permute(0, 2, 3, 1).contiguous().numpy(), 0).argmax(1) == y_all[val_idx]).mean())
        except Exception as e: log('robustness', k, 'failed:', repr(e))
    summary['val_robustness'] = rob; log('val robustness (no TTA):', rob)
summary['test_class_counts'] = np.bincount(sub['label'], minlength=10).tolist()
json.dump(summary, open(os.path.join(RUN, 'summary.json'), 'w'), indent=2)
log('SUMMARY', json.dumps(summary))
