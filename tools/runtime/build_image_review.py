#!/usr/bin/env python3
"""Build a private, randomized image-output review packet.

The generated HTML, copied screenshot inputs, and private arm mapping must stay
under an ignored output directory. The browser-facing packet never names an
engine or marks which output is the frozen reference.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from reference_common import canonical_json, load_manifest, load_private_fixture, sha256_text

GRADES = ("Excellent", "VeryGood", "Good", "NeedsImprovement", "Fail")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _slot_sources(seed: str, fixture_id: str) -> tuple[str, str]:
    if int(sha256_text(f"{seed}:{fixture_id}"), 16) % 2:
        return ("candidate", "reference")
    return ("reference", "candidate")


def build_review_items(
    manifest: Path,
    private_root: Path,
    candidate_ledger: Path,
    output_dir: Path,
    seed: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_manifest(manifest)
    if any(row["mode"] != "image_to_text" for row in rows):
        raise ValueError("image review requires image_to_text fixtures only")

    candidates: dict[str, dict[str, Any]] = {}
    for candidate in _read_jsonl(candidate_ledger):
        fixture_id = str(candidate.get("fixture_id", ""))
        if not fixture_id or fixture_id in candidates:
            raise ValueError(f"candidate ledger fixture must appear exactly once: {fixture_id}")
        generated = candidate.get("generated_text")
        if not isinstance(generated, str):
            raise ValueError(f"candidate output is not text: {fixture_id}")
        if sha256_text(generated) != candidate.get("output_sha256"):
            raise ValueError(f"candidate output hash mismatch: {fixture_id}")
        candidates[fixture_id] = candidate

    expected_ids = {str(row["fixture_id"]) for row in rows}
    if set(candidates) != expected_ids:
        raise ValueError("candidate ledger fixture inventory does not match manifest")

    media_dir = output_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    for row in rows:
        fixture_id = str(row["fixture_id"])
        fixture = load_private_fixture(row, private_root)
        reference = fixture.get("expected_output")
        if not isinstance(reference, str):
            raise ValueError(f"private fixture expected_output is not text: {fixture_id}")
        if sha256_text(reference) != row["expected"]["output_sha256"]:
            raise ValueError(f"reference output hash mismatch: {fixture_id}")
        image_path = fixture["_resolved_media"].get("image_0")
        if image_path is None:
            raise ValueError(f"image_0 is required for review: {fixture_id}")

        candidate = str(candidates[fixture_id]["generated_text"])
        pair_id = sha256_text(f"image-review:{seed}:{fixture_id}")[:16]
        image_name = f"{pair_id}{image_path.suffix.lower()}"
        shutil.copyfile(image_path, media_dir / image_name)
        sources = _slot_sources(seed, fixture_id)
        texts = {"reference": reference, "candidate": candidate}
        outputs = {"A": texts[sources[0]], "B": texts[sources[1]]}
        items.append(
            {
                "pair_id": pair_id,
                "image": f"media/{image_name}",
                "outputs": outputs,
            }
        )
        mappings.append(
            {
                "pair_id": pair_id,
                "fixture_id": fixture_id,
                "slots": {"A": sources[0], "B": sources[1]},
                "output_sha256": {slot: sha256_text(text) for slot, text in outputs.items()},
            }
        )

    review_id = sha256_text(canonical_json(items))
    mapping = {
        "schema_version": "gemma4-e4b-image-review-map-v1",
        "review_id": review_id,
        "manifest_sha256": sha256_text(manifest.read_text(encoding="utf-8")),
        "candidate_ledger_sha256": sha256_text(candidate_ledger.read_text(encoding="utf-8")),
        "items": mappings,
    }
    return items, mapping


def _render_html(items: list[dict[str, Any]], review_id: str) -> str:
    payload = canonical_json(items).replace("</", "<\\/")
    grade_options = "".join(f'<option value="{grade}">{grade}</option>' for grade in GRADES)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gemma 4 image review</title>
<style>
body {{ font: 15px/1.45 -apple-system, BlinkMacSystemFont, sans-serif; margin: 0 auto; max-width: 1200px; padding: 24px; color: #171717; }}
.case {{ border-top: 1px solid #ccc; padding: 28px 0; }}
img {{ border: 1px solid #bbb; display: block; max-height: 720px; max-width: 100%; object-fit: contain; }}
.outputs {{ display: grid; gap: 18px; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); margin-top: 18px; }}
.output {{ background: #f6f6f6; border-radius: 8px; padding: 14px; }}
pre {{ font: 12px/1.4 ui-monospace, SFMono-Regular, monospace; overflow-wrap: anywhere; white-space: pre-wrap; }}
label {{ display: block; margin: 8px 0; }}
textarea {{ box-sizing: border-box; min-height: 70px; width: 100%; }}
button {{ font: inherit; padding: 10px 14px; }}
</style>
</head>
<body>
<h1>Blinded screenshot-description review</h1>
<p>Grade each output independently for usefulness, factual fidelity, and material omissions. Engine identity and reference position are randomized.</p>
<div id="cases"></div>
<button id="export">Export completed review JSON</button>
<script>
const items = {payload};
const grades = `{grade_options}`;
const root = document.getElementById('cases');
function outputCard(item, slot) {{
  const text = document.createElement('pre');
  text.textContent = item.outputs[slot];
  const card = document.createElement('section');
  card.className = 'output';
  card.innerHTML = `<h3>Output ${{slot}}</h3><label>Grade <select data-field="grade" data-slot="${{slot}}"><option value="">Choose…</option>${{grades}}</select></label><label><input type="checkbox" data-field="material_omission" data-slot="${{slot}}"> Material omission</label><label>Notes<textarea data-field="notes" data-slot="${{slot}}"></textarea></label>`;
  card.insertBefore(text, card.children[1]);
  return card;
}}
for (const item of items) {{
  const section = document.createElement('section');
  section.className = 'case';
  section.dataset.pairId = item.pair_id;
  section.innerHTML = `<h2>Case ${{item.pair_id}}</h2><img src="${{item.image}}" alt="Screenshot for ${{item.pair_id}}"><div class="outputs"></div><label>Preference <select data-field="preference"><option value="">Choose…</option><option>A</option><option>B</option><option>Tie</option></select></label>`;
  const outputs = section.querySelector('.outputs');
  outputs.append(outputCard(item, 'A'), outputCard(item, 'B'));
  root.append(section);
}}
document.getElementById('export').addEventListener('click', () => {{
  const results = [...document.querySelectorAll('.case')].map(section => {{
    const result = {{pair_id: section.dataset.pairId, outputs: {{}}}};
    for (const slot of ['A', 'B']) {{
      result.outputs[slot] = {{
        grade: section.querySelector(`[data-field="grade"][data-slot="${{slot}}"]`).value,
        material_omission: section.querySelector(`[data-field="material_omission"][data-slot="${{slot}}"]`).checked,
        notes: section.querySelector(`[data-field="notes"][data-slot="${{slot}}"]`).value,
      }};
    }}
    result.preference = section.querySelector('[data-field="preference"]').value;
    return result;
  }});
  const blob = new Blob([JSON.stringify({{schema_version: 'gemma4-e4b-image-review-v1', review_id: '{review_id}', results}}, null, 2)], {{type: 'application/json'}});
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = 'gemma4-e4b-image-review.json';
  link.click();
  URL.revokeObjectURL(link.href);
}});
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("tests/runtime/fixtures/image/manifest.jsonl"))
    parser.add_argument("--private-root", type=Path, default=Path("tests/runtime/private"))
    parser.add_argument("--candidate-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runtime-reviews/image-w6-phase0"))
    parser.add_argument("--seed", default="gemma4-e4b-phase0-v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    items, mapping = build_review_items(
        args.manifest.resolve(),
        args.private_root.resolve(),
        args.candidate_ledger.resolve(),
        output_dir,
        args.seed,
    )
    (output_dir / "private-arm-map.json").write_text(canonical_json(mapping) + "\n", encoding="utf-8")
    (output_dir / "review.html").write_text(_render_html(items, mapping["review_id"]), encoding="utf-8")
    print(canonical_json({"items": len(items), "review_id": mapping["review_id"], "output": str(output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
