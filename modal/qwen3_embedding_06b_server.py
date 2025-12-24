import modal

FORCE_BUILD = False  # Set to True to force a new image build
LLAMA_CPP_RELEASE = "b7356"
MINUTES = 60

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"



image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12", force_build=FORCE_BUILD)
    .apt_install("git", "build-essential", "cmake", "curl", "libcurl4-openssl-dev")
    .run_commands("git clone https://github.com/ggerganov/llama.cpp"
                f" && cd llama.cpp && git checkout tags/{LLAMA_CPP_RELEASE}")
    .run_commands(
        "cmake llama.cpp -B llama.cpp/build "
        "-DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON "
    )
    .run_commands(  # this one takes a few minutes!
        "cmake --build llama.cpp/build --config Release -j --clean-first --target llama-server"
    )
    .run_commands("cp llama.cpp/build/bin/llama-* llama.cpp")
    .entrypoint([])  # remove NVIDIA base container entrypoint
)

model_cache = modal.Volume.from_name("qwen3-cache", create_if_missing=True)
cache_dir = "/root/.cache/qwen3"
gguf_model_path=f"{cache_dir}/Qwen3-Embedding-0.6B-GGUFs/Qwen3-Embedding-0.6B-bf16.gguf"
app = modal.App("qwen3-embedding-06b")

@app.function(
    image=image, 
    volumes={cache_dir: model_cache},
    gpu="T4", 
    timeout=30 * MINUTES,
    min_containers=1,
)
@modal.concurrent(max_inputs=100)
@modal.web_server(80, startup_timeout=20 * MINUTES)
def llama_server():
    import subprocess

    command = [
        "/llama.cpp/llama-server",     # Server binary
        "-m", gguf_model_path,
        "-c", "16384",
        "--embedding",
        "--host", "0.0.0.0",
        "--port", "80",
        "--pooling", "last"
    ]
    
    print("🦙 running commmand:", command, sep="\n\t")
    subprocess.Popen(command)
