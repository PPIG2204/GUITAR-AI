import sys
import platform
import importlib
import torch

print("\n" + "="*50)
print("🔍 ENVIRONMENT PRE-FLIGHT CHECK")
print("="*50)

# --- SYSTEM INFO ---
print("\n🖥 SYSTEM")
print(f"OS            : {platform.system()} {platform.release()}")
print(f"Python        : {sys.version.split()[0]}")
print(f"Executable    : {sys.executable}")

# --- GPU / TORCH ---
print("\n🔥 TORCH")
print(f"PyTorch       : {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version  : {torch.version.cuda}")
    print(f"GPU           : {torch.cuda.get_device_name(0)}")

# --- REQUIRED LIBRARIES ---
print("\n📦 REQUIRED LIBRARIES")
REQUIRED = [
    "numpy",
    "scipy",
    "librosa",
    "jams",
    "tqdm",
    "matplotlib",
    "sklearn"
]

missing = []
for lib in REQUIRED:
    try:
        module = importlib.import_module(lib)
        version = getattr(module, "__version__", "unknown")
        print(f"✅ {lib:<12} {version}")
    except ImportError:
        print(f"❌ {lib:<12} NOT INSTALLED")
        missing.append(lib)

# --- SUMMARY ---
print("\n" + "-"*50)
if missing:
    print("❌ SETUP INCOMPLETE")
    print("Missing packages:")
    for m in missing:
        print(f"  - {m}")
    print("\n➡️  Install with:")
    print("conda install -c conda-forge " + " ".join(missing))
else:
    print("✅ ALL REQUIREMENTS SATISFIED")
    print("You are good to go 🚀")

print("="*50 + "\n")
