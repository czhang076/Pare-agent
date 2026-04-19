"""Langfuse emitter for Pare runtime telemetry — W3 deliverable.

Bridges the research-branch ``run_agent`` loop to Langfuse spans, with
``metadata.agent_status`` tagging that drives the W4 upstream RFC.

Wired via ``LoopConfig.telemetry_emit`` (a ~15 LOC hook to be added to
``pare.agent.loop`` on the research branch). Default ``None`` so research
pilots are unaffected.
"""
