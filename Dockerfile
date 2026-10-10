FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN pip install --no-cache-dir uv==0.12.23 && uv sync --frozen --no-dev --no-editable
COPY data ./data
COPY sql ./sql
RUN useradd --uid 10001 --create-home worker && mkdir -p /app/.runtime && chown -R worker:worker /app
USER worker
ENV PATH="/app/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1
EXPOSE 8000
CMD ["uvicorn", "sales_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
