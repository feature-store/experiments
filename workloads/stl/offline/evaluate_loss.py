from sktime.performance_metrics.forecasting import mean_squared_scaled_error
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

def get_loss_per_key(key: int, csv_dir, oracle_dir):
    path = f"{csv_dir}/{key}.csv"

    oracle_residual = pd.read_csv(f"{oracle_dir}/oracle_key_A4Benchmark-TS{key}.csv")[
        "pred_residual"
    ]

    df = pd.read_csv(path)
    print(path)
    residual = df["pred_residual"]
    print("residual", len(residual.tolist()))
    mask = ~np.isnan(residual)
    print("residual", len(residual[mask].tolist()))
    loss = mean_squared_scaled_error(
        y_true=oracle_residual[mask], y_pred=residual[mask], y_train=df["value"]
    )
    loss = {
        "loss": loss,
        "n_fits": df["model_version"].dropna().nunique(),
    }
    return loss



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify experiment config")
    parser.add_argument("--csv-path", type=str)
    parser.add_argument("--oracle-path", type=str)
    args = parser.parse_args()

    raw_data = []
    for key in tqdm(range(1, 101)):
        entry = get_loss_per_key(key, csv_dir=args.csv_path, oracle_dir=args.oracle_path)
        raw_data.append({"key": key, **entry})

    df = pd.DataFrame(raw_data)
    print("loss per n_fits")
    print(df.groupby("n_fits")["loss"].describe())
    print(f"loss per key (sample of 10 out of {len(df)})")
    print(df.groupby("key")["loss"].describe().sample(10))
    df.to_csv("final_results.csv")

