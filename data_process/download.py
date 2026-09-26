from huggingface_hub import list_repo_files, hf_hub_download
from pathlib import Path

repo_id = "LucasFang/FLUX-Reason-6M"
prefix = "Aesthetics-Part01"
out_dir = Path("./external/e1374390/work/raw_data")
out_dir.mkdir(parents=True, exist_ok=True)

# 列出仓库文件，筛选以 prefix 开头且以 .parquet 结尾的文件
files = [f for f in list_repo_files(repo_id, repo_type="dataset") if f.startswith(prefix) and f.lower().endswith(".parquet")]
files = sorted(files)[:20]

if not files:
    print("未找到符合条件的 parquet 文件。请检查 prefix 或 repo_id。")
else:
    print(f"准备下载 {len(files)} 个文件到 {out_dir}")
    for i, fname in enumerate(files, 1):
        print(f"[{i}/{len(files)}] 下载 {fname} ...")
        hf_hub_download(repo_id=repo_id, filename=fname, repo_type="dataset", local_dir=str(out_dir), local_dir_use_symlinks=False)
    print("下载完成。")