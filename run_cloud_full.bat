@echo off
set LOCAL_LLM_BASE_URL=https://api.deepseek.com/v1
set LOCAL_LLM_API_KEY=your_deepseek_api_key_here
set LOCAL_LLM_MODEL=deepseek-chat
python -u -X utf8 -m military_research.cli --scenario data/sample_joint_operation.json --case-bank data/military_case_bank_frozen_seed.jsonl --iterations 10 --sim-runs 50 --seed 7 --hope-theta 0.9 --hope-epsilon 0.005 --disable-writeback --output-dir result/multillm_backend/deepseek-cloud/full_optimized
