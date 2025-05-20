import argparse
from data import download_and_save, preprocess_and_save
from src.train import train

def main():
    parser = argparse.ArgumentParser(description="Run pipeline tasks")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---- DOWNLOAD ----
    download_parser = subparsers.add_parser("download", help="Download raw ticker data")
    download_parser.add_argument("--tickers", nargs="+", required=True, help="List of tickers (e.g., AAPL MSFT)")
    download_parser.add_argument("--interval", default="1d", help="Data interval (default: 1m)")
    download_parser.add_argument("--period", default="10y", help="Data period (default: 1d)")
    download_parser.add_argument("--output_dir", default="data/raw", help="Directory to save raw data")



    # ---- PREPROCESS ----
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess raw files")
    preprocess_parser.add_argument("--raw_dir", default="data/raw")
    preprocess_parser.add_argument("--output_dir", default="data/processed")



    # ---- TRAIN ----
    train_parser = subparsers.add_parser("train", help="Train the ML model")
    train_parser.add_argument("--config", default="config/train.yaml", help="Path to training config")

    args = parser.parse_args()

    if args.command == "download":
        download_and_save(
            tickers=args.tickers,
            interval=args.interval,
            period=args.period,
            output_dir=args.output_dir,
        )

    elif args.command == "preprocess":
        preprocess_and_save(raw_dir=args.raw_dir, output_dir=args.output_dir)

    elif args.command == "train":
        print(f"[TODO] Training model")
        train()


    elif args.command == "evaluate":
        print("[TODO] Evaluation logic")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
