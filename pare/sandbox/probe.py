"""Environment probe to determine safe execution modes."""

import asyncio
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class PreFlightProbe:
    """Probes the repository to decide if Concurrent Battle Royale is safe."""
    
    def __init__(self, repo_dir: Path) -> None:
        self.repo_dir = repo_dir

    async def run(self) -> bool:
        """
        Runs a lightweight heuristic to check if concurrent execution is safe.
        Returns True if safe (Battle Royale), False if unsafe (Sequential fallback).
        """
        logger.info("Running pre-flight environment probe...")
        
        # Heuristic: Check for local SQLite databases or known lock files
        # Unsafe if present and no obvious container orchestration found
        unsafe_indicators = [
            ".sqlite",
            "db.sqlite3",
            "celery.pid",
        ]
        
        for root, dirs, files in os.walk(self.repo_dir):
            if '.git' in root:
                continue
            for f in files:
                if any(ind in f for ind in unsafe_indicators):
                    logger.warning("Probe found potentially unsafe shared state: %s", os.path.join(root, f))
                    return False
        
        logger.info("Probe passed. Environment appears safe for Multiverse concurrency.")
        return True
