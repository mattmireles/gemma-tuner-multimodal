"""MLX side: greedy 64 tokens for the same frozen rows and renders."""
import csv, json, sys
from pathlib import Path
sys.path.insert(0, "/Users/mm/Documents/GitHub/gemma-tuner-multimodal")
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from gemma_tuner.models.common.plan31_input_modes import render_plan31_input
from tools.run_plan32_literal_eval import image_views
from tools.plan33_mlx_vision_lora import VISION_LORA_FILE, apply_vision_lora
D = Path("/Users/mm/Documents/GitHub/gemma-tuner-multimodal/data/datasets/tt-screenshot-plan33-literal-v3-deploy/epoch-1")
csv.field_size_limit(1 << 30)
rows = [r for _, r in zip(range(int(sys.argv[2])), csv.DictReader(open(D / "validation.csv")))]
mp = Path(sys.argv[1]); model, proc = load(str(mp))
lora = (mp / VISION_LORA_FILE).exists() and sys.argv[3] == "lora"
if lora: apply_vision_lora(model, mp)
out = []
for mode in ("full", "image_only"):
    for r in rows:
        views, _ = image_views((D / r["image_path"]).resolve(), Path("/tmp/plan33-parity-views"), "x" + r["id"][:12])
        sel, msgs = render_plan31_input(mode=mode, full_prompt=r["prompt"], system_prompt=r["system_prompt"], full_views=views)
        p = apply_chat_template(proc, model.config, msgs, num_images=len(sel), enable_thinking=False)
        res = generate(model=model, processor=proc, prompt=p, image=sel, max_tokens=64, temperature=0.0, enable_thinking=False, verbose=False)
        out.append({"mode": mode, "id": r["id"], "text": res.text})
print(json.dumps(out))
