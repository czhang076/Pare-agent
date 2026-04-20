"""Build Pare-derived Docker images from swebench base images.

Agent tools need ripgrep inside the container for the ``search`` tool to
be fast. Installing it at container start takes 10–40 s per instance
(``apt-get update`` plus package fetch), which is unacceptable over a
pilot. Instead we build a derived image **once** per instance that FROMs
the swebench base and adds a single ``apt-get install`` layer, tagged
``pare-eval.<iid>:latest``. Subsequent container starts hit docker's
image cache in under a second.

Tier 2 verification still uses the original ``swebench/sweb.eval.*:latest``
tag — :func:`run_tier2_in_container` / :class:`DockerEvalSession` must
stay on the base image so our numbers remain comparable to the official
harness. The derived image is agent-side only.

Offline fallback: if ``apt-get update`` cannot reach the mirrors, we skip
the build and return the base tag. The ``search`` tool then falls through
to its ``grep -rn`` branch — slower but functionally equivalent.
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Any

from pare.sandbox.docker_eval import DockerEvalUnavailable, _spec_for

logger = logging.getLogger(__name__)


PARE_IMAGE_PREFIX = "pare-eval"


_DOCKERFILE = b"""FROM {base}
RUN apt-get update \\
    && apt-get install -y --no-install-recommends ripgrep \\
    && rm -rf /var/lib/apt/lists/*
"""


def derived_tag(instance_id: str) -> str:
    """Return the Pare-derived image tag for ``instance_id``.

    Docker tags disallow ``/`` and uppercase letters in the repository
    portion. We normalise by lowercasing and swapping ``/`` → ``__`` so
    ``astropy/astropy-1234`` maps cleanly to ``pare-eval.astropy__astropy-1234``.
    """
    safe = instance_id.replace("/", "__").lower()
    return f"{PARE_IMAGE_PREFIX}.{safe}:latest"


async def ensure_pare_image(
    instance_id: str,
    *,
    dataset_name: str = "princeton-nlp/SWE-bench_Verified",
    split: str = "test",
    force_rebuild: bool = False,
    allow_offline_fallback: bool = True,
) -> str:
    """Resolve (or build) the Pare-derived image for ``instance_id``.

    Fast path: derived tag already present → return it. Slow path: look up
    the base tag via :func:`_spec_for`, ensure it exists locally (pull if
    not), then run ``docker build`` with a single-layer Dockerfile adding
    ripgrep. Returns the tag actually usable for ``containers.create``.

    On offline failure (apt mirrors unreachable) the fallback returns the
    base tag and logs a warning; the caller's search tool needs a
    grep-based fallback path.
    """
    try:
        import docker  # type: ignore
    except ImportError as e:
        raise DockerEvalUnavailable(
            "docker-py not installed; cannot build pare-eval image"
        ) from e

    client = docker.from_env()
    derived = derived_tag(instance_id)

    if not force_rebuild:
        if _image_exists(client, derived):
            logger.debug("ensure_pare_image: cache hit %s", derived)
            return derived

    # Need to build — resolve base tag first.
    _, base_tag = _spec_for(instance_id, dataset_name, split)

    if not _image_exists(client, base_tag):
        logger.info("pulling base image %s (first time — minutes)", base_tag)
        try:
            await asyncio.to_thread(client.images.pull, base_tag)
        except Exception as e:
            raise DockerEvalUnavailable(
                f"failed to pull base image {base_tag}: {e}"
            ) from e

    dockerfile = _DOCKERFILE.replace(b"{base}", base_tag.encode())
    ctx = _single_file_build_context(dockerfile)

    logger.info("building derived image %s FROM %s", derived, base_tag)
    try:
        await asyncio.to_thread(
            _build_image_sync, client, ctx, derived,
        )
    except Exception as e:
        if allow_offline_fallback:
            logger.warning(
                "derived image build failed (%s) — falling back to base %s; "
                "search tool will use grep -rn",
                e, base_tag,
            )
            return base_tag
        raise DockerEvalUnavailable(
            f"derived image build failed for {instance_id}: {e}"
        ) from e
    return derived


def _image_exists(client: Any, tag: str) -> bool:
    try:
        client.images.get(tag)
        return True
    except Exception:
        return False


def _build_image_sync(client: Any, fileobj: io.BytesIO, tag: str) -> None:
    """Run ``client.images.build`` with streamed logs; raise on failure.

    docker-py's high-level ``images.build`` swallows intermediate errors and
    raises a generic ``BuildError`` only if the final image isn't produced.
    We stream the low-level API instead so the caller can see apt-get
    failure messages in the logger output.
    """
    fileobj.seek(0)
    errors: list[str] = []
    for chunk in client.api.build(
        fileobj=fileobj,
        custom_context=True,
        tag=tag,
        rm=True,
        forcerm=True,
        decode=True,
        pull=False,
    ):
        if "stream" in chunk:
            for line in chunk["stream"].splitlines():
                line = line.strip()
                if line:
                    logger.debug("build %s: %s", tag, line)
        if "error" in chunk:
            errors.append(chunk["error"])
            logger.error("build %s: %s", tag, chunk["error"])
    if errors:
        raise RuntimeError("; ".join(errors))


def _single_file_build_context(dockerfile: bytes) -> io.BytesIO:
    """Pack a Dockerfile into an in-memory tar stream for ``api.build``."""
    import tarfile
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name="Dockerfile")
        info.size = len(dockerfile)
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(dockerfile))
    buf.seek(0)
    return buf
