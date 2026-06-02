# RouteHijack — routing-aware jailbreak for MoE LLMs. Four phases:
#   1 data        → corpora (LLM-LAT pairs, C4, AdvBench, MMLU)
#   2 harvest     → localize safety + harmful experts (one model load)
#   3 routehijack → optimize the universal adversarial suffix
#   4 eval        → ASR + MMLU utility + routing-shift (TESR/THPR) + SAFE/AT-RISK verdict

PY := python
DATA := data
ART := artifacts
CONFIG := configs/base.yaml

install:
	pip install -e .

data:
	$(PY) scripts/00_data.py --data-dir $(DATA)

harvest:
	$(PY) scripts/01_harvest.py --config $(CONFIG)

routehijack:
	$(PY) -u scripts/02_routehijack.py --config $(CONFIG) \
		--n-prompts 16 --n-steps 300 --candidates-per-step 128 \
		--candidate-prompt-subsample 0 --grad-batch-size 8 --candidate-batch-size 128 \
		--early-stop-patience 40

eval:
	$(PY) scripts/03_eval.py --config $(CONFIG)

all: data harvest routehijack eval

.PHONY: install data harvest routehijack eval all clean

clean:
	rm -rf $(ART)
