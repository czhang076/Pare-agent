import asyncio
from pare.tools.bash import BashTool
from pare.tools.base import ToolContext
from pathlib import Path

async def main():
    tool = BashTool()
    ctx = ToolContext(cwd=Path('data'))
    
    commands = [
        "python -c 'print(1/0)'", 
        "pytest --collect-only non_existent.py",
        "grep -n 'foo' missing.py",
        "ls -la /nonexistent",
        "python -c 'x = 1'",
    ]
    
    for cmd in commands:
        res = await tool.execute({'command': cmd, 'timeout': 5}, ctx)
        print(f'COMMAND: {cmd}')
        print(f'SUCCESS: {res.success}')
        print(f'ERROR: {res.error}')
        print(f'OUTPUT:\n{res.output}')
        print('-' * 60)

asyncio.run(main())
