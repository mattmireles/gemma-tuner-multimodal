"""HF reference: greedy 64 tokens for frozen rows, stock and adapter-unmerged (training form)."""
import csv, json, sys, torch, transformers
from pathlib import Path
from peft import PeftModel
sys.path.insert(0, "/Users/mm/Documents/GitHub/gemma-tuner-multimodal")
from gemma_tuner.models.common.collators import IMAGE_VIEW_GLOBAL_PLUS_QUADRANTS, _load_image_as_rgb, build_image_views, apply_image_token_budget_to_processor
from gemma_tuner.models.common.plan31_input_modes import render_plan31_input
from gemma_tuner.models.gemma.base_model_loader import load_base_model_for_gemma
from gemma_tuner.models.gemma.family import GemmaFamily
MID, REV = "google/gemma-4-E2B-it", "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
D = Path("/Users/mm/Documents/GitHub/gemma-tuner-multimodal/data/datasets/tt-screenshot-plan33-literal-v3-deploy/epoch-1")
csv.field_size_limit(1 << 30)
rows = [r for _, r in zip(range(int(sys.argv[2])), csv.DictReader(open(D / "validation.csv")))]
proc = transformers.AutoProcessor.from_pretrained(MID, revision=REV); apply_image_token_budget_to_processor(proc, 280)
model = load_base_model_for_gemma(MID, family=GemmaFamily.GEMMA_4, torch_dtype=torch.bfloat16, attn_implementation="sdpa", revision=REV)
if sys.argv[1] != "stock":
    model = PeftModel.from_pretrained(model, sys.argv[1])
model = model.to("mps").eval()
out = []
for mode in ("full", "image_only"):
    for r in rows:
        views = build_image_views(_load_image_as_rgb(str((D / r["image_path"]).resolve())), IMAGE_VIEW_GLOBAL_PLUS_QUADRANTS)
        _, msgs = render_plan31_input(mode=mode, full_prompt=r["prompt"], system_prompt=r["system_prompt"], full_views=views)
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False).to("mps")
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=64, do_sample=False)
        out.append({"mode": mode, "id": r["id"], "text": proc.tokenizer.decode(g[0, inp["input_ids"].shape[1]:], skip_special_tokens=False)})
print(json.dumps(out))
