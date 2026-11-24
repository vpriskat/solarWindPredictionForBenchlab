# Pipeline image: runs the forecast for one t0.   docker run --rm swp-pipeline 20261124T000000Z > forecast.json
# (The competition image is submission/Dockerfile; it stays frozen.)
FROM public.ecr.aws/ubuntu/ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/src
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates python3 python3-pip && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY src ./src
RUN useradd --create-home --uid 1000 runner && chown -R runner /app
USER runner
ENTRYPOINT ["python3", "-m", "swp.model"]
