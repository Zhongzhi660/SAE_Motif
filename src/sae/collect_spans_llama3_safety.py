#!/usr/bin/env python3
"""
Unified SAE span collection for Llama / Mistral / Qwen target models.

Usage:
  # Original mode (top-k spans per neuron, max-pooled across tokens):
  python collect_spans_llama3_safety.py <GPU_ID> <MODEL_KEY> <SUBGROUP> <TOTAL_GROUP> \
      --data-path PATH --threshold THR [--sae-path PATH] [--sae-layer INT]

  # Per-token SAE activations (for token-order sequence motif):
  python collect_spans_llama3_safety.py <GPU_ID> <MODEL_KEY> <SUBGROUP> <TOTAL_GROUP> \
      --data-path PATH --threshold THR --mode token_sae

  # Dense pre-SAE hidden states (for dense embedding control):
  python collect_spans_llama3_safety.py <GPU_ID> <MODEL_KEY> <SUBGROUP> <TOTAL_GROUP> \
      --data-path PATH --threshold THR --mode dense

Output files (all written under data/sae/{attacker_target}/threshold_{THR}/):
  spans mode   : textspans_{model}_group{N}.tsv      -- NeuronID, TextID, Score, Span
  token_sae    : token_sae_{model}_group{N}.tsv      -- TextID, TokenID, NeuronID, Score
  dense        : dense_{model}_group{N}.npz          -- vecs (n, hidden_dim), text_ids (n,)
"""
import sys
import os
import bisect
import tqdm
import torch as tc
import numpy as np

try:
    from .corpus import CorpusSearchIndex
    from .llm_surgery import switch_mode, mount_function
    from .generator import Generator
    from .autoencoder import load_pretrained
except ImportError:
    from corpus import CorpusSearchIndex
    from llm_surgery import switch_mode, mount_function
    from generator import Generator
    from autoencoder import load_pretrained

# ------------------ GPU setup ------------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1] if len(sys.argv) > 1 else "0"
CACHE_DIR = "./"

# ------------------ Per-model config ------------------
MODEL_CONFIG = {
    "llama": {
        "shift": 31,
        "model_ckpt": "llama3-8b",
        "default_sae": os.environ.get("LLAMA_SAE_PATH", ""),
    },
    "mistral": {
        "shift": 5,
        "model_ckpt": "mistral-7b",
        "default_sae": os.environ.get("MISTRAL_SAE_PATH", ""),
    },
    "qwen": {
        "shift": 14,
        "model_ckpt": "qwen2-7b",
        "default_sae": os.environ.get("QWEN_SAE_PATH", ""),
    },
}


# ------------------ Utilities ------------------
class TopKCollector:
    def __init__(self, topK=3):
        self.TopK = topK
        self.items = []
        self.vals = []

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        for _ in self.items:
            yield _

    def _insert(self, val, item):
        idx = bisect.bisect_left(self.vals, val)
        self.vals.insert(idx, val)
        self.items.insert(idx, item)

    def _remove(self):
        self.vals.pop(0)
        self.items.pop(0)

    def update(self, val, item):
        if len(self) < self.TopK:
            self._insert(val, item)
        elif val > self.vals[0]:
            self._remove()
            self._insert(val, item)


# ------------------ Message parsing (shared) ------------------
def parse_messages(text):
    """Parse raw text into a list of chat messages."""
    text = text.replace("\\n", "\n").replace("\\t", "\t")
    if "Human:" in text or "Assistant:" in text:
        messages = []
        current_role = None
        for line in text.split("\n"):
            if line.startswith("Human:"):
                current_role = "user"
                messages.append({"role": current_role, "content": line[len("Human:"):].strip()})
            elif line.startswith("Assistant:"):
                current_role = "assistant"
                messages.append({"role": current_role, "content": line[len("Assistant:"):].strip()})
            elif current_role is not None and line.strip():
                messages[-1]["content"] += " " + line.strip()
        return messages
    return [{"role": "user", "content": text}]


# ══════════════════════════════════════════════════════════════════════════════
# MODE: spans  (original behaviour — max-pooled activations + text spans)
# ══════════════════════════════════════════════════════════════════════════════
IDX = tc.arange(65536)

def _sae_data_root():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "sae")


def _input_parent(data_path):
    return os.path.basename(os.path.dirname(data_path))


def _output_root(data_path, threshold):
    return os.path.join(_sae_data_root(), _input_parent(data_path), f"threshold_{threshold}")


def parse_threshold_values(single_threshold, threshold_list):
    if threshold_list:
        vals = []
        for part in threshold_list.split(","):
            item = part.strip()
            if item:
                vals.append(float(item))
        if not vals:
            raise ValueError("--thresholds was provided but no valid values were parsed.")
        return vals
    return [float(single_threshold)]


def activations(messages, model, sae, tokenizer, threshold=None, size=32, shift=31):
    """
    Compute neuron activations for one conversation.
    Returns max-pooled per-feature scores and the text span around the max token.
    """
    switch_mode(sae, "train")
    topk, sae.topk = sae.topk, 65536

    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception as e:
        sae.topk = topk
        return {"Neurons": [], "Spans": [], "Scores": []}

    ids = tokenizer(text, return_tensors="pt").input_ids[0].to(model._device)

    try:
        with tc.no_grad():
            model.get_activates(ids)
    except RuntimeError:
        pass

    ids = ids[shift:]
    device = model._device
    IDX_dev = IDX.to(device)

    try:
        act, pos = sae.actvs.squeeze()[shift:].max(dim=0)
    except (IndexError, RuntimeError) as e:
        print(f"[WARN] Activation error: {e}")
        sae.topk = topk
        return {"Neurons": [], "Spans": [], "Scores": []}

    if threshold is None:
        choose = act > float("-inf")
    else:
        choose = act > threshold
    idx = IDX_dev[choose].tolist()
    act = act[choose].float().cpu().tolist()
    pos = pos[choose].cpu().tolist()

    spans = [ids[max(0, p - size):p] for p in pos]
    spans = tokenizer.batch_decode(spans)
    sae.topk = topk
    return {"Neurons": idx, "Spans": spans, "Scores": act}


def collect_text_spans(corpus, sae, generator, tokenizer, model_name, subgroup, ttlgroup,
                       max_collects, thresholds, shift, data_path):
    sae.eval()
    sae.MaskTopK = False
    generator._model.eval()
    switch_mode(sae, "train")
    sae.early_stop = True

    thresholds = [float(t) for t in thresholds]
    thresholds_sorted = sorted(thresholds)
    threshold_keys = [f"{t:g}" for t in thresholds_sorted]
    collectors_map = {
        key: [TopKCollector(max_collects) for _ in range(65536)]
        for key in threshold_keys
    }
    for key in threshold_keys:
        os.makedirs(_output_root(data_path, key), exist_ok=True)
    bar = tqdm.tqdm(total=len(corpus), desc=f"Collecting spans ({model_name})")

    for idx, text in enumerate(corpus):
        bar.update(1)
        if idx % ttlgroup != subgroup:
            continue
        messages = parse_messages(text)
        if not messages:
            print(f"[WARN] Empty message skipped at sample {idx}")
            continue
        try:
            results = activations(messages, generator, sae, tokenizer, threshold=None, shift=shift)
        except Exception as e:
            print(f"[WARN] Error at sample {idx}: {e}")
            continue
        for neuron, span, score in zip(results["Neurons"], results["Spans"], results["Scores"]):
            for thr in thresholds_sorted:
                if score > thr:
                    collectors_map[f"{thr:g}"][neuron].update(score, (neuron, idx, score, span))
                else:
                    break

    for key in threshold_keys:
        out_path = os.path.join(_output_root(data_path, key), f"textspans_{model_name}_group{subgroup}.tsv")
        with open(out_path, "w", encoding="utf8") as f:
            f.write("NeuronID\tTextID\tScore\tSpan\n")
            for c in collectors_map[key]:
                for neuron, idx, score, span in c:
                    span = span.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
                    f.write(f"{neuron}\t{idx}\t{score:.8f}\t{span}\n")
        print(f"[spans] Written to {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MODE: token_sae  (per-token SAE activations for token-order sequence motif)
# ══════════════════════════════════════════════════════════════════════════════

def activations_token(messages, model, sae, tokenizer, threshold, shift=31):
    """
    Return per-token SAE activations above threshold.
    Returns list of (token_id, neuron_id, score) tuples.
    """
    switch_mode(sae, "train")
    topk, sae.topk = sae.topk, 65536

    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        sae.topk = topk
        return []

    ids = tokenizer(text, return_tensors="pt").input_ids[0].to(model._device)

    # Reset sae.actvs so we can detect whether the forward pass actually updated it
    sae.actvs = None
    _fwd_err = None
    try:
        with tc.no_grad():
            # Call model directly without output_hidden_states=True to avoid
            # transformers monkey-patching accumulation bug: when early_stop
            # raises RuntimeError, transformers' cleanup (which only catches
            # TypeError) never runs, leaving an extra wrapper per call.
            model._model(
                ids[:512].unsqueeze(0).to(model._device),
                output_hidden_states=False,
            )
    except RuntimeError as e:
        # early_stop raises RuntimeError intentionally after compute_loss;
        # any other RuntimeError (e.g. OOM) also lands here with sae.actvs=None
        _fwd_err = e

    sae.topk = topk

    if sae.actvs is None:
        # forward pass did not reach the SAE layer — stale cache avoided
        if _fwd_err is not None:
            print(f"[WARN] token_sae forward failed (sae.actvs not updated): {_fwd_err}")
        return []

    try:
        # sae.actvs: (n_tokens, 65536); slice off template prefix
        per_token = sae.actvs.squeeze()[shift:].float().cpu()  # (T, 65536)
    except (IndexError, RuntimeError) as e:
        print(f"[WARN] token_sae activation error: {e}")
        return []

    rows = []
    nonzero = (per_token > threshold).nonzero(as_tuple=False)  # (nnz, 2)
    for tok_id, neuron_id in nonzero.tolist():
        score = per_token[tok_id, neuron_id].item()
        rows.append((int(tok_id), int(neuron_id), score))
    return rows


def collect_token_sae(corpus, sae, generator, tokenizer, model_name, subgroup, ttlgroup,
                      threshold, shift):
    """
    Write per-token SAE activations as TextID, TokenID, NeuronID, Score TSV.
    Only tokens/neurons with score > threshold are written.
    """
    sae.eval()
    sae.MaskTopK = False
    generator._model.eval()
    sae.early_stop = True

    sae_data_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "sae")
    input_parent = os.path.basename(os.path.dirname(args.data_path))
    root = os.path.join(sae_data_root, input_parent, f"threshold_{threshold}")
    os.makedirs(root, exist_ok=True)

    out_path = os.path.join(root, f"token_sae_{model_name}_group{subgroup}.tsv")
    bar = tqdm.tqdm(total=len(corpus), desc=f"Collecting token_sae ({model_name})")

    with open(out_path, "w", encoding="utf8") as f:
        f.write("TextID\tTokenID\tNeuronID\tScore\n")
        for idx, text in enumerate(corpus):
            bar.update(1)
            if idx % ttlgroup != subgroup:
                continue
            messages = parse_messages(text)
            if not messages:
                print(f"[WARN] Empty message skipped at sample {idx}")
                continue
            try:
                rows = activations_token(messages, generator, sae, tokenizer, threshold, shift=shift)
            except Exception as e:
                print(f"[WARN] Error at sample {idx}: {e}")
                continue
            for tok_id, neuron_id, score in rows:
                f.write(f"{idx}\t{tok_id}\t{neuron_id}\t{score:.8f}\n")
            # Periodically release CUDA memory fragments
            if idx % 200 == 0 and idx > 0:
                tc.cuda.empty_cache()

    print(f"[token_sae] Written to {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MODE: dense  (pre-SAE hidden states for dense embedding control)
# ══════════════════════════════════════════════════════════════════════════════

def collect_dense(corpus, sae, generator, tokenizer, model_name, subgroup, ttlgroup,
                  thresholds, shift, layer_idx, data_path):
    """
    Write pre-SAE mean-pooled hidden states as a compressed numpy archive.
    Output: dense_{model}_group{N}.npz  with arrays:
      vecs     — float32 (n_queries, hidden_dim)
      text_ids — int64   (n_queries,)

    Strategy: monitor mode + early_stop=False (full forward pass).
    switch_mode is called ONCE before the loop so that per-sample mode
    switching cannot corrupt the hook state.  Monitor mode returns x unchanged
    (a tensor), so the llm_surgery wrapper's .to() call works correctly.
    early_stop=False avoids RuntimeError which would corrupt CUDA async state
    when device_map="auto" offloads layers to CPU.
    """
    sae.eval()
    sae.MaskTopK = False
    generator._model.eval()

    # --- monitor mode: capture pre-SAE hidden state via sae.monitor callback ---
    # Monitor mode: call_hook calls hook.monitor(x) then returns x unchanged.
    # Setting early_stop=False prevents the RuntimeError that corrupts CUDA state
    # with device_map="auto".  The callback stores x; the wrapper receives x back
    # (a tensor), so its .to() call works correctly.
    store = {}

    def _capture(x):
        h = x.detach().float().cpu()
        if h.dim() == 3:
            h = h.squeeze(0)
        store["h"] = h

    sae.monitor = _capture
    switch_mode(sae, "monitor")
    sae.early_stop = False   # no RuntimeError — full forward pass, fully stable

    threshold_keys = [f"{float(t):g}" for t in thresholds]
    for key in threshold_keys:
        os.makedirs(_output_root(data_path, key), exist_ok=True)

    vecs, text_ids = [], []
    bar = tqdm.tqdm(total=len(corpus), desc=f"Collecting dense ({model_name})")

    for idx, text in enumerate(corpus):
        bar.update(1)
        if idx % ttlgroup != subgroup:
            continue
        messages = parse_messages(text)
        if not messages:
            print(f"[WARN] Empty message skipped at sample {idx}")
            continue

        try:
            text_fmt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
        except Exception as e:
            print(f"[WARN] template error at {idx}: {e}")
            continue

        ids = tokenizer(text_fmt, return_tensors="pt").input_ids[0].to(
            generator._device)

        store.clear()
        try:
            with tc.no_grad():
                generator.get_activates(ids)
        except Exception as e:
            print(f"[WARN] forward error at {idx}: {e}")
            continue

        if "h" not in store:
            print(f"[WARN] dense not captured at sample {idx}")
            continue

        h = store["h"]   # (T, D)
        if h.shape[0] <= shift:
            print(f"[WARN] sequence too short ({h.shape[0]} tokens) at {idx}")
            continue

        vec = h[shift:].mean(dim=0).numpy()   # (D,)
        vecs.append(vec)
        text_ids.append(idx)

    if not vecs:
        print("[WARN] No vectors collected.")
        return

    vecs_arr = np.stack(vecs).astype(np.float32)
    ids_arr = np.array(text_ids, dtype=np.int64)
    for key in threshold_keys:
        out_path = os.path.join(_output_root(data_path, key), f"dense_{model_name}_group{subgroup}.npz")
        np.savez_compressed(out_path, vecs=vecs_arr, text_ids=ids_arr)
        print(f"[dense] Written {len(vecs)} vectors to {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect SAE activations / dense embeddings.")
    parser.add_argument("gpu_id",    type=str, help="CUDA device id(s), e.g., '0' or '0,1'")
    parser.add_argument("model_key", type=str, choices=["mistral", "llama", "qwen"])
    parser.add_argument("subgroup",  type=int, help="This worker's subgroup id (0..TOTAL_GROUP-1)")
    parser.add_argument("ttlgroup",  type=int, help="Total number of groups")
    parser.add_argument("--data-path",  type=str, required=True)
    parser.add_argument("--threshold",  type=float, required=True,
                        help="SAE activation threshold (e.g. 2.0). "
                             "For dense mode, controls output directory only.")
    parser.add_argument("--thresholds", type=str, default=None,
                        help="Optional comma-separated list of thresholds. "
                             "When provided, the model is loaded once and outputs are "
                             "written for each threshold in the list.")
    parser.add_argument("--sae-path",   type=str, default=None)
    parser.add_argument("--sae-layer",  type=int, default=None)
    parser.add_argument("--mode",       type=str, default="spans",
                        choices=["spans", "token_sae", "dense"],
                        help="spans: original top-k span collection (default); "
                             "token_sae: per-token SAE activations (TextID/TokenID/NeuronID/Score); "
                             "dense: pre-SAE mean-pooled hidden states (.npz).")

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    model_key = args.model_key
    subgroup  = args.subgroup
    ttlgroup  = args.ttlgroup
    cfg       = MODEL_CONFIG[model_key]

    sae_source = args.sae_path if args.sae_path is not None else cfg["default_sae"]
    print(f"[SAE]    Loading from: {sae_source}")
    name, layer, sae = load_pretrained(sae_source)

    if args.sae_layer is not None:
        print(f"[SAE]    Overriding layer: {layer} -> {args.sae_layer}")
        layer = args.sae_layer

    model_ckpt = cfg["model_ckpt"]
    shift      = cfg["shift"]
    threshold_values = parse_threshold_values(args.threshold, args.thresholds)
    threshold_note = ",".join(f"{t:g}" for t in threshold_values)
    print(f"[CONFIG] model={model_key}, ckpt={model_ckpt}, shift={shift}, "
          f"threshold={threshold_note}, mode={args.mode}")

    corpus    = CorpusSearchIndex(args.data_path, cache_freq=1000, sampling=None)
    generator = Generator(model_ckpt, device="cuda", dtype="bfloat16")
    tokenizer = generator._tokenizer

    print(f"[HOOK]   Mounting SAE to model='{model_key}', layer={layer}")
    mount_function(generator._model, model_key, int(layer), sae)

    with tc.no_grad():
        if args.mode == "spans":
            collect_text_spans(corpus, sae, generator, tokenizer, model_key,
                               subgroup, ttlgroup,
                               max_collects=6000, thresholds=threshold_values, shift=shift,
                               data_path=args.data_path)
        elif args.mode == "token_sae":
            if len(threshold_values) != 1:
                raise ValueError("token_sae mode currently supports a single threshold only.")
            collect_token_sae(corpus, sae, generator, tokenizer, model_key,
                              subgroup, ttlgroup,
                              threshold=threshold_values[0], shift=shift)
        elif args.mode == "dense":
            collect_dense(corpus, sae, generator, tokenizer, model_key,
                          subgroup, ttlgroup,
                          thresholds=threshold_values, shift=shift,
                          layer_idx=int(layer), data_path=args.data_path)
