@echo off
set LOCAL_LLM_BASE_URL=https://api.deepseek.com/v1
set LOCAL_LLM_API_KEY=REDACTED_DEEPSEEK_KEY
set LOCAL_LLM_MODEL=deepseek-chat
python -u -X utf8 -m military_research.cli --scenario data/sample_joint_operation.json --case-bank data/military_case_bank_frozen_seed.jsonl --iterations 10 --sim-runs 50 --seed 7 --disable-memory --disable-hope --disable-reflection --disable-writeback --output-dir result/multillm_backend/deepseek-cloud/pure_llm
