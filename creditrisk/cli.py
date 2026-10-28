"""python -m creditrisk [--n 20000] [--csv loans.csv --target loan_status]"""
import argparse

from .data import generate_loans, load_csv
from .pipeline import run


def main(argv=None):
    p = argparse.ArgumentParser(prog="creditrisk", description=__doc__)
    p.add_argument("--n", type=int, default=20000, help="synthetic loans to generate")
    p.add_argument("--csv", help="use your own CSV instead of synthetic data")
    p.add_argument("--target", default="default", help="name of 0/1 default column in your CSV")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="outputs")
    a = p.parse_args(argv)
    df = load_csv(a.csv, a.target) if a.csv else generate_loans(a.n, a.seed)
    run(df, a.out, a.seed)


if __name__ == "__main__":
    main()
