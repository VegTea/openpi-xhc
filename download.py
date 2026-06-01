from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Posttraining-RFM-RSS2026/Challenge-phase1-dataset",
    repo_type="dataset",
    local_dir="/inspire/ssd/project/gjjproject/czxs24230043/Challenge-phase1-dataset",
    allow_patterns=[
        "tower-of-hanoi-game/expert-data/**",
    ],
)