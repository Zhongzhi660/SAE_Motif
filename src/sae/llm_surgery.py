import torch
import transformers.models.mistral.modeling_mistral as mistral
import transformers.models.llama.modeling_llama as llama

# Try importing Qwen variants
try:
    import transformers.models.qwen2.modeling_qwen2 as qwen2
except Exception:
    qwen2 = None

try:
    import transformers.models.qwen2_moe.modeling_qwen2_moe as qwen2_moe
except Exception:
    qwen2_moe = None

try:
    import transformers.models.qwen2_5.modeling_qwen2_5 as qwen2_5
except Exception:
    qwen2_5 = None

KEY = "__sae_surgery"


def _make_sae_wrapper(original_forward):
    """
    Wrap a decoder layer's original forward to inject an SAE hook after MLP.

    This approach is transformers-version-agnostic: we call the original forward
    (which handles all API differences internally) and then apply the SAE hook
    to the output hidden states. Works whether the original returns a Tensor
    (new transformers ≥4.50) or a tuple (old transformers).
    """
    def wrapped(self, hidden_states, *args, **kwargs):
        result = original_forward(self, hidden_states, *args, **kwargs)
        if hasattr(self, KEY):
            sae_fn = getattr(self, KEY)
            if sae_fn is not None:
                if isinstance(result, torch.Tensor):
                    # New transformers: decoder layer returns tensor directly
                    result = sae_fn(result.to(torch.float32)).to(result.dtype)
                else:
                    # Old transformers: decoder layer returns tuple (hidden_states, ...)
                    hs = sae_fn(result[0].to(torch.float32)).to(result[0].dtype)
                    result = (hs,) + result[1:]
        return result
    return wrapped


# Collect all target decoder classes grouped by model name
_TARGET_CLASSES = {
    "mistral": [mistral.MistralDecoderLayer],
    "llama": [llama.LlamaDecoderLayer],
    "qwen": [],
}

if qwen2 is not None and hasattr(qwen2, "Qwen2DecoderLayer"):
    _TARGET_CLASSES["qwen"].append(qwen2.Qwen2DecoderLayer)
if qwen2_moe is not None and hasattr(qwen2_moe, "Qwen2MoeDecoderLayer"):
    _TARGET_CLASSES["qwen"].append(qwen2_moe.Qwen2MoeDecoderLayer)
if qwen2_5 is not None and hasattr(qwen2_5, "Qwen2_5DecoderLayer"):
    _TARGET_CLASSES["qwen"].append(qwen2_5.Qwen2_5DecoderLayer)

# Keep ops dict for backward compat with mount_function
ops = {name: (cls_list, None) for name, cls_list in _TARGET_CLASSES.items()}

# Monkey-patch all decoder classes with SAE-aware wrappers
for cls_list in _TARGET_CLASSES.values():
    for target_class in cls_list:
        target_class.forward = _make_sae_wrapper(target_class.forward)


def mount_function(model, name, layer_idx, hook):
    assert layer_idx > 0
    for attr in ["enabled", "monitoring", "computing", "early_stop"]:
        if not hasattr(hook, attr):
            print(f"Hook has no attribute {attr}, setting default.")
            setattr(hook, attr, False)
    for func in ["monitor", "compute_loss", "generate"]:
        if not hasattr(hook, func):
            print(f"Hook has no function {func}, setting default.")
            setattr(hook, func, lambda x: x)

    def call_hook(x):
        if not hook.enabled:
            return x
        if hook.monitoring:
            hook.monitor(x)
            if hook.early_stop:
                raise RuntimeError
            return x
        if hook.computing:
            y = hook.compute_loss(x)
            if hook.early_stop:
                raise RuntimeError
            return y
        return hook.generate(x)

    class_list = _TARGET_CLASSES[name]
    hit = False
    for mod_name, layer in model.named_modules():
        if any(isinstance(layer, cls) for cls in class_list):
            layer_idx -= 1
            if layer_idx == 0:
                setattr(layer, KEY, call_hook)
                print(f"Mounted hook at {mod_name}")
                hit = True
                break
    if not hit:
        raise RuntimeError(
            f"Failed to mount hook: no target layer matched for '{name}'. "
            f"Check model family and layer index."
        )


def switch_mode(hook, mode):
    mode = mode.lower()
    assert mode in {"turnoff", "turnon", "monitor", "train", "generate"}
    if mode == "turnoff":
        hook.enabled = False
        return
    hook.enabled = True
    if mode == "turnon":
        return
    hook.monitoring = mode == "monitor"
    hook.computing = mode == "train"
