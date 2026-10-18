#!/usr/bin/env python3
"""LLM codelength baseline in the style of "Language Modeling Is Compression" (Deletang et al., ICLR 2024).

Reports the ideal codelength sum(-log2 p(token)) of a UTF-8 file under a pretrained Hugging Face LM.
An arithmetic coder driven by the same probabilities gets within a few bytes per file of it (flush), plus a
probability-quantisation loss that each record bounds ("coder_overhead_bound_bits"), so results carry the label
"LLM codelength (arithmetic coding adds < 0.01%)". Nothing is actually coded: real use would also need
bit-identical model evaluation on both sides. The weights are not part of the codelength; each record carries
their bf16 size for total-size accounting.

Context methods (--method):
  window  sliding windows of --window tokens advancing --stride tokens; each window scores only the tokens
          no earlier window scored, so every token after the first window sees >= window-stride tokens of context.
  stream  recurrent models (RWKV): the whole file is one sequence, fed --stream-tokens at a time with the
          recurrent state carried across pieces (unbounded context).
  chunks  the LMIC protocol: independent --chunk-bytes chunks (each boundary moved back to a UTF-8 character
          start), each coded from scratch with no context from earlier chunks.
Every sequence starts with the model's document separator (<|endoftext|> for Qwen2.5 and RWKV-4-Pile), so the
first token is charged too. Tokenization is byte-exact: the tokenizer's normalizer (NFC for both models) is
removed and decode(encode(text)) == text is checked; files that are not valid UTF-8 or do not round-trip are
reported as skipped. bpb = bits / original bytes.

RWKV's WKV recurrence runs through wkv(), a chunk-parallel version of transformers' per-token Python loop
(same maths, checked by --selftest), so RWKV is usable on CPU and MPS.

Run with python -I (it reads downloaded weights). Weights come from the Hugging Face cache in $HF_HOME
(default <repo>/downloads/hf), offline; pass --download to fetch a missing snapshot.
  python -I tools/llm_codelength.py data/enwik8/test.bin --model rwkv-169m --json out.json
  python -I tools/llm_codelength.py --summary results/llm
  python -I tools/llm_codelength.py --selftest
"""
import argparse, glob, hashlib, json, math, os, platform, shlex, sys, time, unicodedata

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL = 'LLM codelength (arithmetic coding adds < 0.01%)'
MODELS = {  # alias -> (HF repo, pinned revision)
    'rwkv-169m': ('RWKV/rwkv-4-169m-pile', '46bdc280eb97b6141d5d51a935e0c4870ecaefcc'),
    'qwen-0.5b': ('Qwen/Qwen2.5-0.5B', '060db6499f32faf8b98477b0a26969ef7d8b9987'),
}
SNAPSHOT_FILES = ['config.json', 'tokenizer.json', 'tokenizer_config.json', '*.safetensors', 'pytorch_model.bin',
                  'generation_config.json', 'special_tokens_map.json', 'LICENSE', 'README.md']
CODER_BITS = 31  # assumed arithmetic-coder frequency precision for the overhead bound
LN2 = math.log(2)
ROWS = 512  # positions per logits slice (bounds the vocab-sized logits buffer)


# ----------------------------------------------------------------------------------------------------- RWKV WKV

def wkv(time_decay, time_first, key, value, state=None, return_state=False, T=32):
    """RWKV-4 WKV with transformers' rwkv_linear_attention_cpu signature and maths, computed T steps at a time.

    Within a block every output is a softmax-weighted average over (carried state, keys so far), built from a
    (T, T) matrix of decayed key logits; the state (num, den, log-scale) is carried between blocks in the same
    stabilised form transformers uses. Replaces a Python loop over tokens, which is too slow for whole files.
    """
    import torch
    B, L, C = key.shape
    w = torch.exp(time_decay.float())  # per-step decay is exp(-w)
    u = time_first.float()
    k, v = key.float(), value.float()
    if state is None:
        a = torch.zeros(B, C, device=k.device); b = torch.zeros_like(a); p = torch.full_like(a, -1e38)
    else:
        a, b, p = (s.float() for s in state)  # real numerator = a * exp(p), denominator = b * exp(p)
    out = torch.empty_like(k)
    ar = torch.arange(T, device=k.device)
    for s in range(0, L, T):
        kc, vc = k[:, s:s + T], v[:, s:s + T]
        n = kc.shape[1]
        i = ar[:n].float()
        lag = (i[:, None] - 1 - i[None, :])  # (t, j): t-1-j, the decay steps of key j when output t is formed
        lg = kc[:, None] - lag[None, :, :, None].clamp(min=0) * w  # (B, t, j, C)
        lg = lg.masked_fill((lag < -1)[None, :, :, None], float('-inf'))  # j > t: future keys
        idx = ar[:n]
        lg[:, idx, idx] = u + kc  # j == t: the current key gets the bonus u instead of decay
        ls = p[:, None] - i[None, :, None] * w  # carried state at output t: p - t*w
        m = torch.maximum(lg.amax(2), ls)
        e, es = torch.exp(lg - m[:, :, None]), torch.exp(ls - m)
        out[:, s:s + n] = ((e * vc[:, None]).sum(2) + es * a[:, None]) / (e.sum(2) + es * b[:, None])
        lk = kc - (n - 1 - i)[None, :, None] * w  # state after the block: keys decayed to its last step
        ps = p - n * w
        p2 = torch.maximum(lk.amax(1), ps)
        ek, e0 = torch.exp(lk - p2[:, None]), torch.exp(ps - p2)
        a, b, p = (ek * vc).sum(1) + e0 * a, ek.sum(1) + e0 * b, p2
    return out.to(key.dtype), ([a, b, p] if return_state or state is not None else None)


# ------------------------------------------------------------------------------------------------ model loading

def snapshot(name, revision=None, download=False):
    """(repo, revision, local snapshot dir) for an alias, HF repo id or local directory."""
    if os.path.isdir(name):
        return name, None, name
    repo, rev = MODELS.get(name, (name, revision))
    rev = revision or rev
    hub = os.path.join(os.environ['HF_HOME'], 'hub', 'models--' + repo.replace('/', '--'))
    if rev is None and os.path.exists(os.path.join(hub, 'refs', 'main')):
        rev = open(os.path.join(hub, 'refs', 'main')).read().strip()
    d = os.path.join(hub, 'snapshots', rev or 'main')
    if not os.path.exists(os.path.join(d, 'config.json')):
        if not download:
            raise SystemExit(f'{repo}@{rev} not in {os.environ["HF_HOME"]}; rerun with --download')
        from huggingface_hub import snapshot_download
        d = snapshot_download(repo, revision=rev, allow_patterns=SNAPSHOT_FILES)
    return repo, rev, d


def sha256_file(path, buf=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def load(name, revision, device, bf16, download):
    """Model (eval, on device), byte-exact tokenizer and a description of both for the records."""
    import torch
    from tokenizers import Tokenizer, normalizers
    from transformers import AutoModelForCausalLM
    repo, rev, d = snapshot(name, revision, download)
    tok = Tokenizer.from_file(os.path.join(d, 'tokenizer.json'))
    norm = tok.normalizer
    tok.normalizer = normalizers.Sequence([])  # NFC would make decode(encode(text)) != text for non-NFC input
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16 if bf16 else torch.float32).eval().to(device)
    if m.config.model_type == 'rwkv':
        import transformers.models.rwkv.modeling_rwkv as mr
        mr.rwkv_linear_attention = wkv
    params = sum(p.numel() for p in m.parameters())  # parameters() yields tied tensors once
    wfiles = sorted(glob.glob(os.path.join(d, '*.safetensors')) + glob.glob(os.path.join(d, 'pytorch_model*.bin')))
    sep = m.config.bos_token_id if m.config.bos_token_id is not None else m.config.eos_token_id
    info = dict(name=name, repo=repo, revision=rev, type=m.config.model_type, params=params,
                bytes_bf16=2 * params, vocab=m.get_output_embeddings().weight.shape[0],
                weights={os.path.basename(f): sha256_file(f) for f in wfiles},
                tokenizer_sha256=sha256_file(os.path.join(d, 'tokenizer.json')),
                normalizer_removed=type(norm).__name__ if norm is not None else None,
                separator=None if sep is None else dict(id=sep, text=tok.id_to_token(sep)),
                load_seconds=round(time.time() - t0, 2))
    return m, tok, info


# ------------------------------------------------------------------------------------------------------ scoring

def hidden(m, x, state=None, keep_state=False):
    """Final hidden states (B, L, H) for token ids x, and the carried state of a recurrent model."""
    if m.config.model_type == 'rwkv':
        o = m.rwkv(input_ids=x, state=state, use_cache=keep_state)
        return o.last_hidden_state, o.state
    return m.base_model(input_ids=x, use_cache=False).last_hidden_state, None


def head_bits(m, h, t):
    """-log2 p(t) for hidden rows h (M, H) and targets t (M,), as a float64 CPU tensor; logits in fp32."""
    import torch
    import torch.nn.functional as F
    head = m.get_output_embeddings()
    if getattr(m, '_head32', None) is None:  # fp32 copy of a bf16 head (same tensor when already fp32)
        m._head32 = (head.weight.float(), None if head.bias is None else head.bias.float())
    W, bias = m._head32
    out = []
    for i in range(0, len(h), ROWS):
        z = F.linear(h[i:i + ROWS].float(), W, bias)
        out.append((torch.logsumexp(z, -1) - z.gather(1, t[i:i + ROWS, None])[:, 0]).cpu().double())
    return (torch.cat(out) if out else torch.zeros(0, dtype=torch.float64)) / LN2


def score_batch(m, rows, dev):
    """rows: list of (inp, tgt, lo, hi) token lists; bits of tgt[lo:hi] given inp, one total per row.
    Rows are right-padded: harmless for causal models, the padding is never scored nor attended to."""
    import torch
    L = max(len(r[0]) for r in rows)
    x = torch.zeros(len(rows), L, dtype=torch.long)
    y = torch.zeros_like(x)
    lo = torch.tensor([r[2] for r in rows]); hi = torch.tensor([r[3] for r in rows])
    for b, (inp, tgt, _, _) in enumerate(rows):
        x[b, :len(inp)] = torch.tensor(inp); y[b, :len(tgt)] = torch.tensor(tgt)
    ar = torch.arange(L)
    mask = (ar >= lo[:, None]) & (ar < hi[:, None])
    h, _ = hidden(m, x.to(dev))
    bits = head_bits(m, h[mask.to(dev)], y[mask].to(dev))
    row = torch.arange(len(rows))[:, None].expand(-1, L)[mask]
    return torch.zeros(len(rows), dtype=torch.float64).index_add_(0, row, bits)


def with_sep(ids, sep, vocab):
    """Inputs/targets aligned (tgt[j] is predicted from inp[:j+1]) plus the bits charged outside the model:
    with a separator every token is predicted; without one the first token costs log2(vocab)."""
    if sep is not None:
        ids = [sep] + ids
        return ids[:-1], ids[1:], 0.0
    return ids[:-1], ids[1:], math.log2(vocab)


def run_window(m, ids, sep, vocab, W, S, batch, dev, deadline):
    inp, tgt, extra = with_sep(ids, sep, vocab)
    n, wins, done, s = len(inp), [], 0, 0
    while done < n:  # window [s, s+W) scores the targets [done, e) not scored before
        e = min(s + W, n)
        wins.append((s, e, done)); done = e; s += S
    bits, scored, fwd = extra, 0, 0
    for i in range(0, len(wins), batch):
        part = wins[i:i + batch]
        rows = [(inp[s:e], tgt[s:e], lo - s, e - s) for s, e, lo in part]
        bits += float(score_batch(m, rows, dev).sum())
        scored = part[-1][1]; fwd += sum(e - s for s, e, _ in part)
        if time.time() > deadline:
            break
    return bits, scored + (sep is None and scored > 0), fwd, dict(windows=len(wins))


def run_stream(m, ids, sep, vocab, P, dev, deadline):
    import torch
    if m.config.model_type != 'rwkv':
        raise SystemExit('--method stream needs a recurrent model (RWKV)')
    inp, tgt, extra = with_sep(ids, sep, vocab)
    bits, state, s = extra, None, 0
    for s in range(0, len(inp), P):
        h, state = hidden(m, torch.tensor([inp[s:s + P]], device=dev), state, keep_state=True)
        bits += float(head_bits(m, h[0], torch.tensor(tgt[s:s + P], device=dev)).sum())
        if time.time() > deadline:
            break
    scored = min(s + P, len(inp))
    return bits, scored + (sep is None and scored > 0), scored, dict(pieces=math.ceil(len(inp) / P))


def utf8_cut(data, n):
    """Largest cut <= n that does not split a UTF-8 character."""
    n = min(n, len(data))
    while 0 < n < len(data) and data[n] & 0xC0 == 0x80:
        n -= 1
    return n


def run_chunks(m, tok, data, sep, vocab, CB, batch, dev, deadline):
    cuts, pos = [0], 0
    while pos < len(data):
        pos = max(utf8_cut(data, pos + CB), pos + 1)
        cuts.append(pos)
    chunks, moved = [], 0
    for a, b in zip(cuts, cuts[1:]):
        t = data[a:b].decode('utf-8')
        ids = tok.encode(t, add_special_tokens=False).ids
        if tok.decode(ids, skip_special_tokens=False) != t:
            raise ValueError(f'chunk at byte {a} does not round-trip through the tokenizer')
        chunks.append((b - a, ids)); moved += (b - a) != CB and b != len(data)
    order = sorted(range(len(chunks)), key=lambda c: -len(chunks[c][1]))  # length-sorted batches pad less
    bits, nbytes, ntok, fwd, done, rounded = 0.0, 0, 0, 0, 0, 0
    for i in range(0, len(order), batch):
        rows, extra = [], 0.0
        for c in order[i:i + batch]:
            inp, tgt, extra = with_sep(chunks[c][1], sep, vocab)
            rows.append((inp, tgt, 0, len(tgt))); nbytes += chunks[c][0]; ntok += len(chunks[c][1])
        per = score_batch(m, rows, dev) + extra
        bits += float(per.sum()); rounded += sum(math.ceil(b / 8) for b in per.tolist())
        fwd += sum(len(r[0]) for r in rows); done += len(rows)
        if time.time() > deadline:
            break
    return bits, ntok, fwd, dict(chunks=len(chunks), chunks_scored=done, boundaries_moved_for_utf8=moved,
                                 bytes_scored=nbytes, codelength_bytes_whole_bytes_per_chunk=rounded)


def codelength(m, tok, info, path, a, dev):
    """One record for one file (or a 'skipped' record)."""
    import torch
    raw = open(path, 'rb').read()
    rec = dict(label=LABEL, file=os.path.relpath(os.path.abspath(path), REPO), file_bytes=len(raw),
               sha256=hashlib.sha256(raw).hexdigest())
    data = raw[:utf8_cut(raw, a.limit_bytes)] if a.limit_bytes else raw
    if a.limit_bytes:
        rec['slice_sha256'] = hashlib.sha256(data).hexdigest()
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError as e:
        return dict(rec, skipped=f'not valid UTF-8: {e}')
    if not data:
        return dict(rec, skipped='empty')
    enc = tok.encode(text, add_special_tokens=False)
    if tok.decode(enc.ids, skip_special_tokens=False) != text:
        return dict(rec, skipped='tokenizer does not round-trip this text byte-exactly')
    sep = info['separator']['id'] if info['separator'] else None
    vocab = info['vocab']
    method = a.method or ('stream' if info['type'] == 'rwkv' else 'window')
    deadline = time.time() + 60 * a.max_minutes if a.max_minutes else float('inf')
    sync = (lambda: torch.mps.synchronize()) if dev == 'mps' else (lambda: None)
    sync(); t0 = time.time()
    with torch.inference_mode():
        if method == 'window':
            ctx = dict(window_tokens=a.window, stride_tokens=a.stride, batch=a.batch)
            bits, ntok, fwd, more = run_window(m, enc.ids, sep, vocab, a.window, a.stride, a.batch, dev, deadline)
        elif method == 'stream':
            ctx = dict(stream_tokens=a.stream_tokens)
            bits, ntok, fwd, more = run_stream(m, enc.ids, sep, vocab, a.stream_tokens, dev, deadline)
        else:
            ctx = dict(chunk_bytes=a.chunk_bytes, batch=a.batch, context_across_chunks=False)
            bits, ntok, fwd, more = run_chunks(m, tok, data, sep, vocab, a.chunk_bytes, a.batch, dev, deadline)
    sync(); secs = time.time() - t0
    if method == 'chunks':
        nbytes = more.pop('bytes_scored')
        complete, seqs = more['chunks_scored'] == more['chunks'], more['chunks_scored']
    else:
        complete, seqs = ntok == len(enc.ids), 1
        nbytes = len(data) if complete else len(text[:enc.offsets[ntok - 1][1]].encode('utf-8'))
    ctx.update(more)
    # coder overhead estimate: a 31-bit frequency table loses <= 2*vocab/2^31 nats per token to flooring every
    # frequency to >= 1 and rounding, and each independently coded sequence needs <= 32 bits to flush
    overhead = ntok * 2 * vocab / 2 ** CODER_BITS / LN2 + 32 * seqs
    rel = overhead / bits
    label = LABEL if rel < 1e-4 else f'LLM codelength (arithmetic coding adds up to {rel:.2%}: flush per sequence)'
    return dict(rec, label=label, model=info, method=method, context=ctx, complete=complete, bytes=nbytes,
                nfc_stable=unicodedata.normalize('NFC', text) == text, tokens=ntok,
                bytes_per_token=round(nbytes / max(ntok, 1), 4), bits=round(bits, 3), bpb=round(bits / nbytes, 6),
                codelength_bytes=math.ceil(bits / 8), total_with_model_bf16_bytes=math.ceil(bits / 8) + info['bytes_bf16'],
                coder_overhead_est_bits=round(overhead, 1), coder_overhead_est_rel=round(rel, 7),
                device=dev, dtype='bf16' if a.bf16 else 'fp32', threads=torch.get_num_threads(),
                seconds=round(secs, 2), tokens_per_s=round(ntok / secs, 1), forward_tokens_per_s=round(fwd / secs, 1))


def versions():
    import numpy, tokenizers, torch, transformers
    return dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
                tokenizers=tokenizers.__version__, numpy=numpy.__version__, platform=platform.platform(),
                machine=platform.machine())


# ------------------------------------------------------------------------------------------------------ summary

def summary(d):
    """Collect <d>/runs/*.json into <d>/summary.json and a markdown table <d>/summary.md."""
    rows = []
    for f in sorted(glob.glob(os.path.join(d, 'runs', '*.json'))):
        j = json.load(open(f))
        for r in j['runs']:
            rows.append(dict(source=os.path.relpath(f, REPO), **r))
    pub = {}
    sota = os.path.join(REPO, 'results', 'published', 'sota.json')
    if os.path.exists(sota):
        pub = {r['id']: r for r in json.load(open(sota))['records']}
    ref = [dict(id=k, method=pub[k]['method'], dataset=pub[k]['dataset'], bpb=pub[k]['bpb'], params=pub[k]['params'])
           for k in ('ts_zip-enwik8', 'deletang-chinchilla-1b', 'deletang-llama-2-7b',
                     'deletang-transformer-3-2m-trained-on-enwik8') if k in pub]
    json.dump(dict(label=LABEL, generated=time.strftime('%Y-%m-%d %H:%M:%S'), runs=rows, published=ref),
              open(os.path.join(d, 'summary.json'), 'w'), indent=1)
    md = [f'# {LABEL}: results table', '',
          f'Generated by `python -I tools/llm_codelength.py --summary {os.path.relpath(d, REPO)}` from `runs/*.json`. '
          'bpb = bits / original bytes; "+model" adds the bf16 weights (2 bytes/parameter).', '',
          '| file | model | method | context | bytes | tokens | bpb | codelength bytes | +model bf16 bytes | device | tok/s | complete |',
          '|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---|']
    for r in rows:
        if 'skipped' in r:
            md.append(f"| {r['file']} | | | skipped: {r['skipped']} | {r['file_bytes']:,} | | | | | | | |")
            continue
        c = r['context']
        ctx = (f"W{c['window_tokens']}/S{c['stride_tokens']}" if r['method'] == 'window' else
               f"{c['chunk_bytes']} B chunks, no history" if r['method'] == 'chunks' else 'state carried')
        md.append(f"| {r['file']} | {r['model']['name']} | {r['method']} | {ctx} | {r['bytes']:,} | {r['tokens']:,} | "
                  f"{r['bpb']:.4f} | {r['codelength_bytes']:,} | {r['total_with_model_bf16_bytes']:,} | "
                  f"{r['device']} {r['dtype']} | {r['tokens_per_s']:,.0f} | {'yes' if r['complete'] else 'NO'} |")
    if ref:
        md += ['', 'Published reference points (from `results/published/sota.json`; different data and protocols):', '',
               '| source | dataset | bpb | params |', '|---|---|---:|---|']
        md += [f"| {r['method']} | {r['dataset']} | {r['bpb']} | {r['params']} |" for r in ref]
    open(os.path.join(d, 'summary.md'), 'w').write('\n'.join(md) + '\n')
    print(f'{len(rows)} records -> {d}/summary.json, {d}/summary.md')


# ----------------------------------------------------------------------------------------------------- selftest

def selftest():
    """CPU checks with tiny random models (no weights, no network): wkv() against transformers' loop, the three
    context methods against each other and a plain full-logits computation, and the skip paths."""
    import tempfile, torch
    import transformers.models.rwkv.modeling_rwkv as mr
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import Qwen2Config, Qwen2ForCausalLM, RwkvConfig, RwkvForCausalLM
    torch.manual_seed(0)
    ref_loop = mr.rwkv_linear_attention_cpu
    B, L, C = 2, 45, 8
    td, tf = torch.randn(C) * 2, torch.randn(C)
    k, v = torch.randn(B, L, C) * 3, torch.randn(B, L, C)
    o1, s1 = ref_loop(td, tf, k, v, return_state=True)
    o2, s2 = wkv(td, tf, k, v, return_state=True, T=8)
    o3, _ = ref_loop(td, tf, k[:, :7], v[:, :7], state=[x.clone() for x in s1])
    o4, _ = wkv(td, tf, k[:, :7], v[:, :7], state=s2, T=4)
    err = max((o1 - o2).abs().max(), (o3 - o4).abs().max(), *((x - y).abs().max() / (y.abs().max() + 1) for x, y in zip(s1, s2)))
    assert err < 1e-4, f'wkv mismatch {err}'
    print(f'wkv vs transformers loop: max err {float(err):.2e}')

    alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocab = {'<|endoftext|>': 0, **{ch: i + 1 for i, ch in enumerate(sorted(alphabet))}}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    V = len(vocab)
    tiny = {'rwkv': RwkvForCausalLM(RwkvConfig(vocab_size=V, hidden_size=32, num_hidden_layers=2, attention_hidden_size=32,
                                               intermediate_size=64, context_length=64, bos_token_id=0, eos_token_id=0)),
            'qwen2': Qwen2ForCausalLM(Qwen2Config(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                                  num_attention_heads=4, num_key_value_heads=2, bos_token_id=0, eos_token_id=0))}
    text = 'def f(x):\n    return x * 2  # été ☃\n' * 3
    data = text.encode()
    ids = tok.encode(text).ids
    inf = float('inf')
    for kind, m in tiny.items():
        m.eval()
        with torch.inference_mode():
            full = torch.tensor([[0] + ids])
            mr.rwkv_linear_attention = ref_loop  # reference through transformers' own per-token loop
            lp = torch.log_softmax(m(full).logits[0, :-1].float(), -1)
            mr.rwkv_linear_attention = wkv
            ref = float(-lp.gather(1, full[0, 1:, None]).sum() / LN2)
            one = run_window(m, ids, 0, V, 4096, 4096, 1, 'cpu', inf)[0]
            ch = run_chunks(m, tok, data, 0, V, 10 ** 6, 1, 'cpu', inf)[0]
            win = run_window(m, ids, 0, V, 32, 16, 3, 'cpu', inf)[0]
            win1 = run_window(m, ids, 0, V, 32, 16, 1, 'cpu', inf)[0]
            short = run_chunks(m, tok, data, 0, V, 7, 4, 'cpu', inf)
            assert abs(one - ref) < 1e-3 * ref and abs(ch - ref) < 1e-3 * ref, (kind, one, ch, ref)
            assert abs(win - win1) < 1e-4 * ref, (kind, win, win1)  # batching/padding does not change results
            assert short[3]['chunks'] >= len(data) // 7 and short[3]['bytes_scored'] == len(data)
            msg = f'{kind}: full {ref:.3f} bits, one window {one:.3f}, one chunk {ch:.3f}, W32/S16 {win:.3f}'
            if kind == 'rwkv':
                st = run_stream(m, ids, 0, V, 5, 'cpu', inf)[0]
                assert abs(st - ref) < 1e-3 * ref, (st, ref)
                msg += f', stream {st:.3f}'
            print(msg)
    with tempfile.NamedTemporaryFile(suffix='.bin') as f:
        f.write(b'ok \xff'); f.flush()
        a = argparse.Namespace(limit_bytes=0)
        r = codelength(None, tok, dict(separator=None, vocab=V), f.name, a, 'cpu')
        assert r['skipped'].startswith('not valid UTF-8'), r
    print('selftest ok')


# ---------------------------------------------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('files', nargs='*')
    ap.add_argument('--model', default='rwkv-169m', help=f'alias ({", ".join(MODELS)}), HF repo id or local dir')
    ap.add_argument('--revision', help='HF revision (aliases are pinned)')
    ap.add_argument('--method', choices=['window', 'stream', 'chunks'], help='default: stream for RWKV, else window')
    ap.add_argument('--window', type=int, default=2048, help='window tokens (window method)')
    ap.add_argument('--stride', type=int, default=1024, help='window advance in tokens (window method)')
    ap.add_argument('--stream-tokens', type=int, default=1024, help='tokens per forward, state carried (stream)')
    ap.add_argument('--chunk-bytes', type=int, default=2048, help='independent chunk size (chunks method, LMIC)')
    ap.add_argument('--batch', type=int, default=1, help='windows/chunks per forward')
    ap.add_argument('--limit-bytes', type=int, default=0, help='score only the first N bytes (validation slices)')
    ap.add_argument('--max-minutes', type=float, default=0, help='stop a file after this long (record says complete=false)')
    ap.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--bf16', action='store_true', help='bf16 weights/activations (default fp32; logits always fp32)')
    ap.add_argument('--download', action='store_true', help='fetch a missing snapshot from the Hugging Face Hub')
    ap.add_argument('--json', help='write the records here')
    ap.add_argument('--summary', metavar='DIR', help='collect DIR/runs/*.json into DIR/summary.{json,md} and exit')
    ap.add_argument('--selftest', action='store_true', help='run the CPU self-checks and exit')
    a = ap.parse_args()
    os.environ.setdefault('HF_HOME', os.path.join(REPO, 'downloads', 'hf'))
    if not a.download:
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('RAYON_NUM_THREADS', str(a.threads))  # tokenizers' thread pool
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'true')
    if a.summary:
        return summary(a.summary)
    import torch
    torch.set_num_threads(a.threads)
    if a.selftest:
        return selftest()
    if not a.files:
        ap.error('no input files')
    if a.device == 'mps' and not torch.backends.mps.is_available():
        raise SystemExit('MPS not available')
    m, tok, info = load(a.model, a.revision, a.device, a.bf16, a.download)
    runs, cmd = [], ' '.join(shlex.quote(x) for x in [sys.executable, '-I'] + sys.argv)
    for f in a.files:
        try:
            r = codelength(m, tok, info, f, a, a.device)
        except Exception as e:  # e.g. out of memory on one file: record it and go on with the others
            r = dict(file=os.path.relpath(os.path.abspath(f), REPO), file_bytes=os.path.getsize(f),
                     sha256=sha256_file(f), skipped=f'error: {type(e).__name__}: {e}'[:500])
        runs.append(r)
        if 'skipped' in r:
            print(f"{r['file']}: skipped ({r['skipped']})", flush=True)
        else:
            print(f"{r['file']}: {r['model']['name']} {r['method']} {r['bytes']:,} B {r['tokens']:,} tok "
                  f"{r['bits'] / 8:,.0f} B  {r['bpb']:.4f} bpb  {r['seconds']:.0f} s  {r['tokens_per_s']:,.0f} tok/s"
                  + ('' if r['complete'] else '  INCOMPLETE'), flush=True)
        if a.json:  # rewritten after every file, so an interrupted run keeps what it finished
            os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
            json.dump(dict(label=LABEL, command=cmd, cwd=os.getcwd(), date=time.strftime('%Y-%m-%d %H:%M:%S %z'),
                           versions=versions(), runs=runs), open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
