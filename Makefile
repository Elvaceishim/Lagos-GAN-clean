.PHONY: test build compose-up compose-down docker-build

test:
    python -m pytest -q

build:
    docker build -t lagos-gan-inference:latest -f Dockerfile.prod .

compose-up:
    docker-compose up -d --build

compose-down:
    docker-compose down

docker-build:
    docker build -t lagos-gan-inference:latest .