from pathlib import Path
from datetime import datetime, timezone
import hashlib, importlib.metadata as md, json, platform, time
import torch
from diffusers import AutoPipelineForText2Image

MODEL_ID = "stabilityai/sd-turbo"
REVISION = "b261bac6fd2cf515557d5d0707481eafa0485ec2"
PROMPT = (
    "abstract illustration for a digital archive concept, "
    "a glowing network of interconnected data nodes and light streams, "
    "deep blue and soft cyan palette, "
    "clean minimal composition, no text, no logos, no people"
)
SEED, STEPS, GUIDANCE, SIZE = 20260917, 1, 0.0, 512

out = Path("artifacts/run_001")
out.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
load_args = dict(revision=REVISION, use_safetensors=True, torch_dtype=dtype)
if device == "cuda":
    load_args["variant"] = "fp16"

started = time.perf_counter()
pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID, **load_args)
pipe = pipe.to(device)
print("Device:", device)
print("CUDA available:", torch.cuda.is_available())
generator = torch.Generator(device="cpu").manual_seed(SEED)
image = pipe(
    prompt=PROMPT,
    num_inference_steps=STEPS,
    guidance_scale=GUIDANCE,
    height=SIZE,
    width=SIZE,
    generator=generator,
).images[0]
elapsed = time.perf_counter() - started

image_path = out / "result.png"
image.save(image_path)
sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()

packages = {name: md.version(name) for name in ["torch", "diffusers", "transformers", "accelerate", "safetensors"]}
manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "model_id": MODEL_ID,
    "revision": REVISION,
    "prompt": PROMPT,
    "seed": SEED,
    "steps": STEPS,
    "guidance_scale": GUIDANCE,
    "height": SIZE,
    "width": SIZE,
    "device": device,
    "dtype": str(dtype),
    "platform": platform.platform(),
    "packages": packages,
    "elapsed_seconds_measured": elapsed,
    "artifact": str(image_path),
    "sha256": sha256,
}
(out / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
print(json.dumps(manifest, ensure_ascii=False, indent=2))