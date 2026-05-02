"""
Compare OmniVoice's internal LM weights to stock Qwen3-0.6B.

If they're nearly identical → OmniVoice didn't fine-tune the LM, safe to swap.
If they diverge significantly → swapping a quantized Qwen3 will hurt quality.

Run inside the container:
    docker compose -f docker-compose.gpu.yaml exec omnivoice-backend python compare_lm_weights.py
"""
import torch
from omnivoice import OmniVoice
from transformers import AutoModelForCausalLM

print("Loading OmniVoice...")
omni = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda",
    dtype=torch.bfloat16,
    load_asr=False,
)
omni_lm = omni.llm
print(f"OmniVoice LM: {type(omni_lm).__name__}")

print("\nLoading stock Qwen3-0.6B (instruct)...")
stock = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)
print(f"Stock LM: {type(stock).__name__}")

print("\nComparing parameters...")
omni_params = dict(omni_lm.named_parameters())
# Strip "model." prefix from stock (it's Qwen3ForCausalLM wrapping Qwen3Model)
stock_raw = dict(stock.named_parameters())
stock_params = {}
for k, v in stock_raw.items():
    if k.startswith("model."):
        stock_params[k[len("model."):]] = v
    elif k == "lm_head.weight":
        continue  # OmniVoice doesn't have lm_head
    else:
        stock_params[k] = v

shared = set(omni_params) & set(stock_params)
only_omni = set(omni_params) - set(stock_params)
only_stock = set(stock_params) - set(omni_params)

print(f"  Shared param names: {len(shared)}")
print(f"  Only in OmniVoice LM: {len(only_omni)}")
print(f"  Only in stock Qwen3: {len(only_stock)}")

if only_omni:
    print(f"  First few only-omni: {list(only_omni)[:5]}")
if only_stock:
    print(f"  First few only-stock: {list(only_stock)[:5]}")

# Quick sanity: print first 5 values of layer 0 q_proj for both
sanity = "layers.0.self_attn.q_proj.weight"
if sanity in omni_params and sanity in stock_params:
    a = omni_params[sanity].float().flatten()[:5]
    b = stock_params[sanity].float().flatten()[:5]
    print(f"\nSanity check — first 5 values of {sanity}:")
    print(f"  omni:  {a.tolist()}")
    print(f"  stock: {b.tolist()}")

print("\nL2 difference per shared parameter (top 10 most different):")
diffs = []
for name in shared:
    a = omni_params[name].float()
    b = stock_params[name].float()
    if a.shape != b.shape:
        print(f"  SHAPE MISMATCH on {name}: {a.shape} vs {b.shape}")
        continue
    rel_diff = (a - b).norm() / (b.norm() + 1e-8)
    diffs.append((name, rel_diff.item(), a.shape))

diffs.sort(key=lambda x: -x[1])
for name, rel, shape in diffs[:10]:
    print(f"  {rel*100:6.2f}%  {name}  {tuple(shape)}")

print(f"\nMean relative diff across {len(diffs)} shared params: "
      f"{sum(d[1] for d in diffs) / max(len(diffs), 1) * 100:.3f}%")

print("\nInterpretation:")
print("  <0.1%  → essentially identical, safe to swap")
print("  <2%    → minor finetuning, swap probably fine")
print("  >5%    → significant finetuning, swap will hurt quality")
