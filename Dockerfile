# Pipeline image: runs the Prefect live cycle for a t0 (or default latest cycle).
# Example:
#   docker run --rm swp-pipeline 20261126T000000Z
#   docker run --rm -e SWP_STORAGE_URL="file:///data/store" -v /tmp/swp_store:/data/store swp-pipeline

FROM public.ecr.aws/ubuntu/ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    SWP_STORAGE_URL=""

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

COPY src ./src
COPY flow ./flow

RUN useradd --create-home --uid 1000 runner && chown -R runner /app
USER runner

ENTRYPOINT ["python3", "flow/cli.py"]