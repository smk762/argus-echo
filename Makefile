UV ?= uv

# Which engines `make launch`, `rebuild` and `logs` act on: compose profiles (miso, xtts, qwen, whisper, wyoming-*).
#   make launch ENGINES="xtts qwen"
# Empty, compose uses COMPOSE_PROFILES from the environment or .env; set, ENGINES wins over it.
ENGINES ?=
COMPOSE = docker compose $(foreach engine,$(ENGINES),--profile $(engine))

.PHONY: help install lint format test build clean launch rebuild stop status logs engines

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Create venv and install with test extras
	$(UV) venv
	$(UV) pip install -e ".[dev]"

lint:  ## Ruff check + format check
	$(UV)x ruff@0.15.16 check src/ tests/
	$(UV)x ruff@0.15.16 format --check src/ tests/

format:  ## Ruff autoformat + autofix
	$(UV)x ruff@0.15.16 format src/ tests/
	$(UV)x ruff@0.15.16 check --fix src/ tests/

test:  ## Run the test suite
	$(UV) run --no-sync pytest --tb=short -q

build:  ## Build sdist + wheel
	$(UV) build

clean:  ## Remove build + cache artifacts
	rm -rf dist build .venv *.egg-info .pytest_cache .ruff_cache src/argus_echo/_version.py

engines:
	@if [ -z "$(ENGINES)" ] && [ -z "$$COMPOSE_PROFILES" ] && ! grep -qsE '^COMPOSE_PROFILES=.' .env; then \
		echo 'No engine chosen: every engine is a compose profile, so compose starts nothing by default.'; \
		echo '  make $(or $(MAKECMDGOALS),launch) ENGINES="xtts qwen"     # this once'; \
		echo '  COMPOSE_PROFILES=xtts,qwen in .env      # from now on (plain "docker compose up -d" works too)'; \
		echo 'Engines: miso (~17 GB VRAM), xtts (~2 GB), qwen, whisper (~3 GB), wyoming-xtts, wyoming-miso, wyoming-qwen, wyoming-whisper'; \
		exit 1; \
	fi

launch: engines  ## Start the engines: ENGINES="xtts qwen", or COMPOSE_PROFILES in .env
	$(COMPOSE) up -d

rebuild: engines  ## Build their images from this checkout, then start them
	$(COMPOSE) up -d --build

logs: engines  ## Follow their logs
	$(COMPOSE) logs -f --tail 100

status:  ## Every argus-echo container, running or not
	docker compose --profile '*' ps -a

stop:  ## Stop and remove every argus-echo container
	docker compose --profile '*' down
