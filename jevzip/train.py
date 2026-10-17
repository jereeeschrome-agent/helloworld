"""Train the two-stream model on a byte corpus (wall-clock budgeted; CPU, Apple-silicon MPS or CUDA)."""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import PAD, Config, JevZip
from .schedule import reveal_plan

LN2 = math.log(2)


def make_batch(data, cfg: Config, bs, rng, p_noctx=0.2, p_trunc=0.05, rounds=None, kmin=8, kmax=128):
    """Random windows, each with a random schedule; every block byte is a supervised slot."""
    C, B = cfg.ctx, cfg.block
    L = C + B
    ctok = np.full((bs, L), PAD, np.int64)
    cpos = np.zeros((bs, L), np.int64)
    qpos = np.zeros((bs, B), np.int64)
    qvis = np.full((bs, B), C, np.int64)
    tgt = np.full((bs, B), -100, np.int64)
    cpos[:, :C] = np.arange(C)
    for i, o in enumerate(rng.integers(0, len(data) - L, bs)):
        win = data[o : o + L]
        n = int(rng.integers(1, B)) if rng.random() < p_trunc else B
        k = rounds or int(round(math.exp(rng.uniform(math.log(kmin), math.log(kmax)))))
        order, bounds = reveal_plan(cfg.order, B, n, k, cfg.first)
        bounds = np.asarray(bounds)
        if rng.random() >= p_noctx:
            ctok[i, :C] = win[:C]
        blk = win[C:][order]
        ctok[i, C : C + n] = blk
        cpos[i, C : C + n] = C + order
        group = np.searchsorted(bounds, np.arange(n), side="right") - 1
        qpos[i, :n] = C + order
        qvis[i, :n] = C + bounds[group]
        tgt[i, :n] = blk
    return [torch.from_numpy(a) for a in (ctok, cpos, qpos, qvis, tgt)]


def loss_bits(model, batch, device="cpu", amp=True):
    ctok, cpos, qpos, qvis, tgt = (t.to(device, non_blocking=True) for t in batch)
    with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=amp):
        logits = model(ctok, cpos, qpos, qvis)
    return F.cross_entropy(logits.view(-1, logits.shape[-1]), tgt.view(-1), ignore_index=-100) / LN2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "weights" / "jevzip-custom.pt")
    ap.add_argument("--minutes", type=float, default=100)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", type=Path, help="warm-start from exported weights")
    ap.add_argument("--kmin", type=int, default=8, help="training schedules: K log-uniform in [kmin, kmax]")
    ap.add_argument("--kmax", type=int, default=128)
    ap.add_argument("--order", choices=["bitrev", "lanes", "linear"], default="bitrev", help="linear + --kmin/--kmax 512 = AR control")
    ap.add_argument("--val-rounds", type=int, default=16)
    ap.add_argument("--device", default="cpu", help="cpu, mps (Apple silicon) or cuda")
    ap.add_argument("--no-amp", action="store_true", help="fp32 instead of bf16 autocast")
    size = ap.add_argument_group("model size (fresh models only)")
    for name in ("ctx", "block", "d_model", "n_layers", "n_heads", "ff_mult"):
        size.add_argument("--" + name.replace("_", "-"), dest=name, type=int, default=getattr(Config, name))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.init:
        model = JevZip.load(args.init).train()
        cfg = model.cfg
    else:
        sizes = {k: getattr(args, k) for k in ("ctx", "block", "d_model", "n_layers", "n_heads", "ff_mult")}
        cfg = Config(order=args.order, **sizes)
        model = JevZip(cfg)
    model.to(args.device)
    amp = not args.no_amp
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M on {args.device} {cfg}", flush=True)
    train = np.memmap(args.data / "train.bin", dtype=np.uint8, mode="r")
    val = np.memmap(args.data / "val.bin", dtype=np.uint8, mode="r")
    val_batches = [make_batch(val, cfg, 16, np.random.default_rng(1000 + i), p_noctx=0, p_trunc=0, rounds=args.val_rounds) for i in range(4)]

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "pos" not in n and "tok" not in n]
    other = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "pos" not in n and "tok" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": other, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))

    budget = args.minutes * 60
    start = time.time()
    step, ema = 0, None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    while True:
        frac = (time.time() - start) / budget
        if frac >= 1:
            break
        lr = args.lr * min(1, (step + 1) / args.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for group in opt.param_groups:
            group["lr"] = lr
        loss = loss_bits(model, make_batch(train, cfg, args.bs, rng, kmin=args.kmin, kmax=args.kmax), args.device, amp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        step += 1
        if step % 50 == 0:
            print(f"step {step} t={time.time() - start:.0f}s lr={lr:.2e} train {ema:.3f} bpb", flush=True)
        if step % 500 == 0:
            evaluate(model, val_batches, step, args.device, amp)
            torch.save(model.export(), args.out)
    evaluate(model, val_batches, step, args.device, amp)
    torch.save(model.export(), args.out)
    print(f"saved {args.out}", flush=True)


@torch.no_grad()
def evaluate(model, batches, step, device="cpu", amp=True):
    model.eval()
    bpb = sum(loss_bits(model, b, device, amp).item() for b in batches) / len(batches)
    model.train()
    print(f"step {step} VAL {bpb:.3f} bpb", flush=True)


if __name__ == "__main__":
    main()
