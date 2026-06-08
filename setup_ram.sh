# setup_ram.sh — put model weights, activation shards, and all outputs on the
# location with the MOST free space, instead of assuming /dev/shm or a fixed size.
#
# It scans a set of candidate roots (RAM disk + common disk volumes), measures
# free space with `df`, and points HF cache + data/ cache/ artifacts/ at the
# biggest writable one. A big disk volume wins over a small RAM disk automatically,
# which is what large models (e.g. Qwen3-235B ≈ 470 GB) need — a RAM disk can't
# hold them. On a tiny-disk pod, the RAM disk still wins, preserving the old behavior.
#
# USE IT (must be sourced so the exports stick):
#   cd /workspace/routehijack
#   source ./setup_ram.sh

# Grow /dev/shm toward ~80% of RAM so the RAM disk competes fairly as a candidate
# (leaving headroom for the process). Best-effort — ignored if not permitted.
mount -o remount,size=80% /dev/shm 2>/dev/null || true

# Candidate storage roots: existing + writable; the one with the most free space wins.
_cands="/dev/shm /tmp /workspace /scratch /data /mnt /local $HOME $(pwd)"

_best=""; _best_avail=0; _best_type=""
for _d in $_cands; do
  [ -d "$_d" ] && [ -w "$_d" ] || continue
  _avail=$(df -Pk "$_d" 2>/dev/null | awk 'NR==2{print $4}')   # 1K-block Available
  case "$_avail" in ''|*[!0-9]*) continue ;; esac              # numeric only
  if [ "$_avail" -gt "$_best_avail" ]; then
    _best_avail="$_avail"; _best="$_d"
    _best_type=$(df -PTk "$_d" 2>/dev/null | awk 'NR==2{print $2}')
  fi
done

if [ -z "$_best" ]; then
  echo "setup_ram: no writable candidate found among: $_cands — leaving paths as-is."
else
  _root="$_best/routehijack-scratch"
  mkdir -p "$_root/hf_cache"
  export HF_HOME="$_root/hf_cache"
  export TRANSFORMERS_CACHE="$_root/hf_cache"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TOKENIZERS_PARALLELISM=false

  # Symlink data/ cache/ artifacts/ at the scratch root WITHOUT destroying existing
  # data — so re-sourcing (or a spot restart on the same volume) keeps checkpoints,
  # cached harvest sweeps, and results for --resume.
  for d in data cache artifacts; do
    mkdir -p "$_root/$d"
    if [ -L "$d" ] || [ ! -e "$d" ]; then
      rm -f "$d"; ln -sfn "$_root/$d" "$d"
    elif [ -d "$d" ] && [ -z "$(ls -A "$d" 2>/dev/null)" ]; then
      rmdir "$d"; ln -sfn "$_root/$d" "$d"
    else
      echo "  ! ./$d is a non-empty real dir — leaving it as-is (move it aside to use $_root/$d)"
    fi
  done

  _avail_h=$(df -Ph "$_best" 2>/dev/null | awk 'NR==2{print $4}')
  echo "setup_ram: chose $_best (${_best_type:-?}, ${_avail_h:-?} free) -> $_root"
  echo "  HF_HOME=$HF_HOME ; data/ cache/ artifacts/ -> $_root"
  case "$_best_type" in
    tmpfs|ramfs)
      echo "  WARNING: this is a RAM disk — WIPED on stop/preemption. On SPOT, mount a"
      echo "  persistent volume (so it wins by free space) before sourcing, or your"
      echo "  weights + checkpoints + --resume state won't survive a restart."
      ;;
  esac
fi
