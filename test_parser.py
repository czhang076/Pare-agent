import json, glob, os

files = glob.glob("data/swebench_workdirs/.pare_sessions/pylint-dev__pylint-7080_session_*.jsonl")
latest_session = max(files, key=os.path.getmtime)
print(f"Reading {latest_session}...\n")
with open(latest_session) as f:
    events = [json.loads(line) for line in f]

for i, e in enumerate(events):
    if e["type"] == "tool_call":
        d = e["data"]
        print(f"\n[{i}] CALL {d.get('tool')}: {json.dumps(d.get('params', {}), ensure_ascii=False)}")
    elif e["type"] == "tool_result":
        d = e["data"]
        print(f"[{i}] RESULT (success={d.get('success')}, error={d.get('error')})")
        output = d.get("output", "")
        if output:
            lines = output.splitlines()
            if len(lines) > 20:
                print("OUTPUT:")
                print("\n".join(lines[:10]))
                print("... [snip] ...")
                print("\n".join(lines[-10:]))
            else:
                print(f"OUTPUT:\n{output}")
        else:
            print("OUTPUT: <empty>")
        print("-" * 80)
