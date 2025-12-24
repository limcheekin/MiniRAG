# Usage: 
# modal run --detach download_qwen3.py::download_qwen3_coder
# modal run --detach download_qwen3.py::download_qwen3_next_instruct
# modal run --detach download_qwen3.py::download_qwen3_4b_instruct
# modal run --detach download_qwen3.py::download_qwen3_4b_thinking
# modal run --detach download_qwen3.py::download_qwen3_30b_a3b_instruct
# modal run --detach download_qwen3.py::download_qwen3_embedding_06b

import modal

FORCE_BUILD = False  # Set to True to force a new image build

app = modal.App("download-models")
model_cache = modal.Volume.from_name("qwen3-cache", create_if_missing=True)
cache_dir = "/root/.cache/qwen3"

image = (
    modal.Image.debian_slim(force_build=FORCE_BUILD)
    .apt_install("git-lfs")
)

@app.function(
    image=image, volumes={cache_dir: model_cache}, timeout=7200 * 2 # 120 x 2 minutes timeout to handle potential long downloads
)
def download_qwen3_coder():
    import os

    os.chdir(cache_dir)
    os.system('git clone https://huggingface.co/Intel/Qwen3-Coder-30B-A3B-Instruct-gguf-q2ks-mixed-AutoRound')
    model_cache.commit()
    
@app.function(
    image=image, volumes={cache_dir: model_cache}, timeout=7200 * 2 # 120 x 2 minutes timeout to handle potential long downloads
)
def download_qwen3_next_instruct():
    import os

    os.chdir(cache_dir)
    os.system('git clone https://huggingface.co/Intel/Qwen3-Next-80B-A3B-Instruct-int4-mixed-AutoRound')
    model_cache.commit()
    
@app.function(
    image=image, volumes={cache_dir: model_cache}, timeout=7200
)
def download_qwen3_4b_instruct():
    import os

    os.chdir(cache_dir)
    os.system('GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF')
    os.chdir("Qwen3-4B-Instruct-2507-GGUF")
    os.system('git lfs pull --include="Qwen3-4B-Instruct-2507-Q4_K_M.gguf"')
    model_cache.commit()
    
@app.function(
    image=image, volumes={cache_dir: model_cache}, timeout=7200
)
def download_qwen3_4b_thinking():
    import os

    os.chdir(cache_dir)
    os.system('GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/unsloth/Qwen3-4B-Thinking-2507-GGUF')
    os.chdir("Qwen3-4B-Thinking-2507-GGUF")
    os.system('git lfs pull --include="Qwen3-4B-Thinking-2507-Q4_K_M.gguf"')
    model_cache.commit()
    
@app.function(
    image=image, volumes={cache_dir: model_cache}, timeout=7200
)
def download_qwen3_14b():
    import os

    os.chdir(cache_dir)
    os.system('GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Qwen/Qwen3-14B-GGUF')
    os.chdir("Qwen3-14B-GGUF")
    os.system('git lfs pull --include="Qwen3-14B-Q4_K_M.gguf"')
    model_cache.commit()

@app.function(
    image=image, volumes={cache_dir: model_cache}, timeout=7200
)
def download_qwen3_30b_a3b_instruct():
    import os

    os.chdir(cache_dir)
    os.system('GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF')
    os.chdir("Qwen3-30B-A3B-Instruct-2507-GGUF")
    os.system('git lfs pull --include="Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"')
    model_cache.commit()

@app.function(
    image=image, volumes={cache_dir: model_cache}, timeout=3600
)
def download_qwen3_embedding_06b():
    import os

    os.chdir(cache_dir)
    os.system('GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Casual-Autopsy/Qwen3-Embedding-0.6B-GGUFs')
    os.chdir("Qwen3-Embedding-0.6B-GGUFs")
    os.system('git lfs pull --include="Qwen3-Embedding-0.6B-bf16.gguf"')
    model_cache.commit()
