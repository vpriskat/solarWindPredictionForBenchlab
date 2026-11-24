.PHONY: install test lint docker run replay
install:
	pip install -e ".[dev,image]"
test:
	pytest -q
lint:
	ruff check src tests scripts
docker:
	docker build --platform linux/amd64 -t swp-pipeline .
run:            # make run T0=20261124T000000Z
	docker run --rm swp-pipeline $(T0)
replay:         # replays past cycles through the image and scores them against NOAA's file
	python3 scripts/replay_eval.py --days 10 --image swp-pipeline --jobs 3
