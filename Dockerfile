FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app
COPY pyproject.toml .python-version uv.lock ./
RUN uv sync --locked

RUN mkdir models
COPY models/hgb_regressor_0.5_500.pkl models/hgb_regressor_0.5_500.pkl
COPY models/one_hot_encoder.pkl models/one_hot_encoder.pkl
COPY src/service.py service.py

CMD ["uv", "run", "uvicorn", "--host", "0.0.0.0", "service:app"]
