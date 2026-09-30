import wandb, subprocess, json
commit = subprocess.check_output(["git","rev-parse","HEAD"], cwd="/Users/vein/Documents/research/aspect-persistence").decode().strip()
run = wandb.init(
    entity="sugam-panthi-university-of-southern-mississippi",
    project="lapse",
    group="mem0-scale",
    name="mem0-scale-gemini-3-flash-2026-09-22",
    id="mem0-scale-gemini-3-flash-2026-09-22",
    resume="allow",
    config={
        "writer": "google/gemini-3-flash-preview",
        "writer_provider": "OpenRouter (mem0 openai provider, host not pinnable)",
        "mem0_version": "2.0.19 OSS, unmodified write path",
        "embedder": "fastembed BAAI/bge-small-en-v1.5",
        "vector_store": "qdrant on-disk, isolated per shard/user",
        "inputs_sha256": "91c4df1fe69cf8355857b7c549888687543ef5b6f6a191553f70f6ba1703b595",
        "n_items": 512,
        "forms": ["prog", "simple", "simple_fornow", "simple_atm"],
        "n_per_form": 128,
        "temperature": 0.0,
        "seed": None,
        "git_commit": commit,
        "predecessor_run": "exploratory/mem0-scale-2026-09-12 (writer deepseek/deepseek-v4-flash)",
        "spec": "research/MEM0_SCALE_GEMINI_SPEC_2026-09-22.md",
    },
)
print("run url:", run.url)
run.log({"status": "launched", "shards": 4})
