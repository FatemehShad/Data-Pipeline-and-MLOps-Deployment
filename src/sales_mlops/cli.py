import argparse
import json
from pathlib import Path

from sales_mlops.pipeline import ingest
from sales_mlops.training import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["pipeline", "train", "upload"])
    parser.add_argument("--database", type=Path, default=Path(".runtime/warehouse.db"))
    parser.add_argument("--orders", type=Path, default=Path("data/orders.csv"))
    parser.add_argument("--products", type=Path, default=Path("data/products.json"))
    parser.add_argument("--sql", type=Path, default=Path("sql/transform.sql"))
    parser.add_argument("--output", type=Path, default=Path(".runtime/reports"))
    parser.add_argument("--max-mae", type=float, default=15.0)
    parser.add_argument("--bucket")
    parser.add_argument("--prefix", default="sales/")
    args = parser.parse_args()
    if args.command == "pipeline":
        data = ingest(args.orders, args.products, args.database, args.sql)
        print(json.dumps({"rows": len(data), "database": str(args.database)}))
    elif args.command == "train":
        print(json.dumps(train(args.database, args.output, args.max_mae)))
    else:
        if not args.bucket:
            parser.error("upload requires --bucket")
        import boto3

        s3 = boto3.client("s3")
        for path in (args.database, args.output / "evaluation.json"):
            s3.upload_file(
                str(path),
                args.bucket,
                args.prefix + path.name,
                ExtraArgs={"ServerSideEncryption": "AES256"},
            )
        print("Uploaded warehouse and evaluation report")


if __name__ == "__main__":
    main()
