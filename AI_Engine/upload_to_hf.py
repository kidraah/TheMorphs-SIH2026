"""
Upload SIH26077 Hybrid TransUNet to Hugging Face Hub.

Run AFTER training completes:
    python upload_to_hf.py

What it uploads:
  - checkpoints/best_model.pth  (trained weights)
  - model.py                    (architecture — needed to load the model)
  - config.py                   (normalization stats, hyperparams)
  - data_loader.py              (dataset structure reference)
  - README.md                   (auto-becomes the Hugging Face model card)
"""

from huggingface_hub import HfApi, create_repo
import os

# ── CONFIGURE THESE ────────────────────────────────────────────────────────
HF_USERNAME  = ""   # Leave empty to be prompted during runtime
REPO_NAME    = "SIH26077-HybridTransUNet"
# ──────────────────────────────────────────────────────────────────────────

# Since this script is now inside scripts/, the project root is the parent directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES_TO_UPLOAD = [
    # (local path,                           path inside HF repo)
    ("checkpoints/best_model.pth",           "best_model.pth"),
    ("model.py",                             "model.py"),
    ("config.py",                            "config.py"),
    ("data_loader.py",                       "data_loader.py"),
    ("README.md",                            "README.md"),
]

def main():
    global HF_USERNAME
    if not HF_USERNAME or HF_USERNAME == "YOUR_HUGGINGFACE_USERNAME":
        HF_USERNAME = input("Enter your Hugging Face username: ").strip()
        if not HF_USERNAME:
            print("[!] Username cannot be empty. Aborting.")
            return

    REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"
    api = HfApi()

    # Create the repo (safe to run if it already exists)
    print(f"Creating / verifying repo: {REPO_ID}")
    create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True, private=False)
    print("  [OK] Repo ready")

    # Upload each file
    for local_rel, repo_path in FILES_TO_UPLOAD:
        local_abs = os.path.join(PROJECT_ROOT, local_rel)
        if not os.path.exists(local_abs):
            print(f"  [SKIP] {local_rel} not found — skipping")
            continue

        size_mb = os.path.getsize(local_abs) / 1024**2
        print(f"  Uploading {local_rel}  ({size_mb:.1f} MB) → {repo_path} ...")
        api.upload_file(
            path_or_fileobj=local_abs,
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type="model",
        )
        print(f"    [OK] {repo_path}")

    print(f"\n{'='*60}")
    print(f"  Upload complete!")
    print(f"  Model page: https://huggingface.co/{REPO_ID}")
    print(f"{'='*60}")
    print()
    print("  Anyone can now load your model with:")
    print(f"    from huggingface_hub import hf_hub_download")
    print(f"    ckpt = hf_hub_download('{REPO_ID}', 'best_model.pth')")


if __name__ == "__main__":
    main()
