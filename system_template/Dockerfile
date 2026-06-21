# Reproducible final-evaluation image, built with uv (staff manual §6.4).
FROM python:3.11-slim

# Bring in the uv binary. Pin to a specific version for full reproducibility.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Use the image's interpreter; never silently download another one.
ENV UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install deps first (better layer caching). A team that commits a uv.lock gets
# reproducible builds; without one, uv resolves at build time.
COPY pyproject.toml ./
RUN uv sync --no-dev

# Put the project venv on PATH so the run contract stays `python run.py ...`.
ENV PATH="/app/.venv/bin:$PATH"

COPY . .

# Contract: read /input, write /output.
#   docker run --rm -v $PWD/input:/input:ro -v $PWD/output:/output \
#     <image> python run.py --input /input/inputs.jsonl --output /output/predictions.jsonl
