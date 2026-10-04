# Autovision training image (CUDA runtime base).
FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime AS runtime

ENV VTRAIN_CACHE_DIR=/home/autovision/.cache/autovision \
    PIP_NO_CACHE_DIR=1
WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
RUN pip install -e ".[dev]" && python -m compileall -q src/autovision

# Smoke test: synthetic images -> pretrain -> artifacts asserted.
CMD ["bash", "docker/smoke_test.sh"]
