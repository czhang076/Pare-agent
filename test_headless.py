import sys
with open("pare/cli/headless.py", encoding="utf-8") as f:
    in_func = False
    for line in f:
        if line.startswith("def _build_trajectory_record"):
            in_func = True
        elif in_func and line.startswith("def "):
            break
        
        if in_func:
            sys.stdout.write(line)
