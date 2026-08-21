VENV := /goinfre/$(USER)/call_me_maybe/.venv

UV_ENV := UV_PROJECT_ENVIRONMENT=$(VENV)


install:
	@mkdir -p /goinfre/$(USER)/call_me_maybe
	$(UV_ENV) UV_LINK_MODE=copy uv sync


run:
	$(UV_ENV) uv run python -m src


debug:
	$(UV_ENV) uv run python -m pdb -m src


clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -rf .mypy_cache
	@rm -rf .pytest_cache
	@rm -rf .ruff_cache
	@echo "Temporary files and caches removed."


lint:
	$(UV_ENV) uv run flake8 . --exclude=llm_sdk
	$(UV_ENV) uv run mypy . \
		--exclude='^llm_sdk/' \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs


lint-strict:
	$(UV_ENV) uv run flake8 . --exclude=llm_sdk
	$(UV_ENV) uv run mypy . --exclude='llm_sdk/' --strict


activate:
	@echo "Run:"
	@echo "source $(VENV)/bin/activate"


deactivate:
	@echo "Run:"
	@echo "deactivate"


sync: install


.PHONY: install run debug clean lint lint-strict activate deactivate sync