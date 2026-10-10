# Cloud-Native Data Pipeline and MLOps Deployment

A runnable portfolio project demonstrating Python/SQL ingestion, data validation,
MLflow experiments and model registry, Docker/Kubernetes deployment, AWS integration,
and GitHub Actions. Python 3.12 is required; dependencies are pinned in `uv.lock`.

## Architecture

CSV orders + JSON product catalogue → validation → SQLite staging → SQL join and
revenue calculation → chronological training/evaluation → MLflow versioned model
and `champion` alias → FastAPI prediction service → Prometheus metrics.

The committed data is synthetic and reproducible. The model estimates order revenue
from quantity, price, and category; revenue is their known arithmetic relationship.
This intentionally exercises MLOps plumbing, **not** a meaningful future-demand model
or evidence of business forecasting performance. Replace these sources and target
with real historical data before claiming predictive value.

## Run locally

Install [uv](https://docs.astral.sh/uv/), then from the repository root:

```bash
uv sync --frozen
uv run sales-mlops pipeline
uv run sales-mlops train
uv run uvicorn sales_mlops.api:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
curl --fail http://127.0.0.1:8000/ready
curl --fail -H 'Content-Type: application/json' \
  -d '{"units":3,"unit_price":20,"category":"home"}' \
  http://127.0.0.1:8000/predict
curl --fail http://127.0.0.1:8000/metrics
uv run pytest -q
uv run ruff check .
```

In the Codex cloud machine use `export UV_CACHE_DIR=/workspace/.cache/uv` because
its default home cache is read-only. Docker tooling likewise needs
`export DOCKER_CONFIG=/workspace/.docker-local`. Use the existing isolated checkout;
do not create a Git worktree for routine tasks.

`pipeline` rejects missing columns, nulls, duplicate keys, unknown products, invalid
dates, and nonpositive/nonfinite numbers before publishing a replacement warehouse.
Repeated ingestion does not duplicate rows. Training uses a chronological 80/20
split, seed 42, and logs MAE/RMSE, parameters, model signature, and evaluation report.
A new model version is registered each time; only versions meeting `--max-mae`
(default 15 revenue units) replace `champion`. A rejected candidate exits nonzero.
The API loads its model at startup; restart it after promoting another version.

Runtime files live in ignored `.runtime/`: warehouse, MLflow SQLite database,
model artifacts, and evaluation report. Local paths are absolute in MLflow metadata;
run training and serving in the same root. Moving files alone is not portable.

Optional MLflow UI (bind internally):

```bash
uv run mlflow server --backend-store-uri sqlite:///.runtime/mlflow.db \
  --default-artifact-root "$(pwd)/.runtime/artifacts" --host 127.0.0.1 --port 5000
```

## Docker and monitoring

```bash
docker compose up --build -d --wait api prometheus
docker compose logs
docker compose down
```

Compose trains before starting the API and retains registry/artifacts in a named
volume. API uses port 8000; Prometheus scrapes it every 15 seconds. Monitor request
counts/errors, inference latency, and prediction distributions. Ground-truth accuracy
and drift detection in production need later outcomes; these are not implemented.
`docker compose down -v` removes the generated data and models.

On a cloud runner whose Docker bridge cannot resolve package hosts, build using
`docker build --network=host -t sales-mlops:local .` if supported, then
`docker compose up --no-build -d --wait api`.

## Kubernetes and AWS

`deploy/k8s/app.yaml` provides a namespace, PVC, training init container, one API
replica, probes, resource limits and internal service. Substitute `sales-mlops:local`
with your registry image before applying. The PVC must use a working default storage
class. This demonstration retrains on each rollout and uses SQLite plus a single
ReadWriteOnce volume with a Recreate rollout. For multiple replicas use an external
MLflow service with PostgreSQL and S3 artifact storage and separate training jobs.

The manual `.github/workflows/deploy.yml` publishes a SHA-tagged image to ECR,
applies the manifest to an existing EKS cluster, waits for readiness, and checks a
prediction through port-forwarding. It does not create AWS resources. Configure
these GitHub **production environment variables**:

- `AWS_REGION`, `ECR_REPOSITORY`, `EKS_CLUSTER`
- `AWS_DEPLOY_ROLE_ARN`: an IAM role trusted for GitHub OIDC restricted to this
  repository and the production environment. Enable production environment reviews.

Provision ECR, EKS and a default EBS CSI storage class first. The role needs scoped
ECR push permissions, `eks:DescribeCluster`, and EKS access/RBAC to manage these
namespace resources. No static AWS keys are stored. Trigger the workflow on `main`
with the confirmation input `deploy`. The CI workflow runs tests, lint, training,
container startup and a prediction on pushes and pull requests.

Optional S3 publication using an existing AWS role/profile:

```bash
uv run sales-mlops upload --bucket YOUR_BUCKET --prefix sales/
```

This uploads the warehouse and evaluation report with server-side encryption. It
does not upload the MLflow registry or models. The role needs scoped `s3:PutObject`;
KMS buckets may require compatible encryption settings. Use workload roles or your
secure environment settings for authentication, never keys committed to source.

The API and local MLflow store are demonstration components. Add authentication,
TLS at ingress, backups, and production data governance before public deployment.
AWS deployment incurs resource costs and has not been performed by local setup.
