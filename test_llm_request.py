import json, glob, os

files = glob.glob("data/swebench_workdirs/.pare_sessions/pylint-dev__pylint-7080_session_*.jsonl")
latest_session = max(files, key=os.path.getmtime)
with open(latest_session) as f:
    events = [json.loads(line) for line in f]

for i, e in enumerate(events):
    if e["type"] == "llm_request":
        print(f"\n[{i}] LLM_REQUEST")
        d = e["data"]
        # print first 500 chars to see if messages are fully logged
        print(str(d)[:500])
        print("-" * 80)
