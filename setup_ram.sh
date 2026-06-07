# setup_ram.sh — run EVERYTHING in RAM (disk is too small for the model).
#
# Puts the model weights, activation shards, and all outputs in /dev/shm (RAM),
# which is bigger than the disk here. Works because the model loads into GPU VRAM,
# so CPU RAM is free to act as the disk.
#
# RAM is wiped when the pod stops/restarts — that's fine, you just re-source this
# and the model re-downloads. Nothing crashes from a full disk.
#
# USE IT (must be sourced so the exports stick):
#   cd /workspace/moe-misuse-probe
#   source ./setup_ram.sh

mount -o remount,size=26G /dev/shm 2>/dev/null || true   # grow RAM disk (26 of 32 GB; rest for the process)

export HF_HOME=/dev/shm/hf_cache                          # model weights -> RAM
export TRANSFORMERS_CACHE=/dev/shm/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p /dev/shm/hf_cache

for d in data cache artifacts; do                         # shards + outputs -> RAM (relative paths in the repo)
  rm -rf "$d"; mkdir -p "/dev/shm/$d"; ln -sfn "/dev/shm/$d" "$d"
done

echo "RAM disk:"; df -h /dev/shm
echo "HF_HOME=$HF_HOME ; data/ cache/ artifacts/ -> /dev/shm"
