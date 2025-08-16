import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    errors = y_true - y_pred
    return float(np.sqrt(np.mean(errors ** 2)))


def compute_top_feature_importances(
    train_csv_path: str = os.path.join("artifacts", "train.csv"),
    model_path: str = os.path.join("artifacts", "model.pkl"),
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl"),
    sample_size: int = 2000,
    n_repeats: int = 3,
    top_k: int = 20,
) -> List[Tuple[str, float]]:
    """
    Compute permutation feature importance on original input features and
    return top-k most important ones.

    This is model-agnostic and respects the saved preprocessing pipeline,
    so importances are attributed to the original input columns.

    Returns a list of (feature_name, importance_in_rmse_delta).
    """
    try:
        if not os.path.exists(train_csv_path):
            raise FileNotFoundError(f"Train CSV not found at {train_csv_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")

        # Load artifacts
        model = load_object(model_path)
        preprocessor = load_object(preprocessor_path)

        # Load training data (so columns match exactly what the preprocessor expects)
        df = pd.read_csv(train_csv_path)

        # Prepare X (original features) and y in dollar space (business metric)
        id_column = "Id"
        target_column = "SalePrice"
        feature_df = df.drop(columns=[c for c in [id_column, target_column] if c in df.columns], errors="ignore")
        if target_column not in df.columns:
            raise ValueError("Target column 'SalePrice' was not found in training data.")
        y_dollars = df[target_column].values.astype(float)

        # Optional down-sample for faster computation
        if sample_size is not None and len(feature_df) > sample_size:
            sampled_idx = feature_df.sample(n=sample_size, random_state=42).index
            feature_df = feature_df.loc[sampled_idx].reset_index(drop=True)
            y_dollars = y_dollars[sampled_idx]

        # Baseline performance
        X_trans = preprocessor.transform(feature_df)
        y_pred_log_base = model.predict(X_trans)
        # Convert model outputs (log1p scale) back to dollars to reflect business metric
        y_pred_base = np.expm1(y_pred_log_base)
        baseline_rmse = _rmse(y_dollars, y_pred_base)

        feature_names = list(feature_df.columns)
        importances = np.zeros(len(feature_names), dtype=float)

        rng = np.random.default_rng(42)

        for repeat_idx in range(n_repeats):
            for idx, col in enumerate(feature_names):
                original_values = feature_df[col].to_numpy(copy=True)

                # Permute this single column
                shuffled = original_values.copy()
                rng.shuffle(shuffled)
                feature_df[col] = shuffled

                # Recompute predictions with permuted column
                X_perm = preprocessor.transform(feature_df)
                y_pred_log_perm = model.predict(X_perm)
                y_pred_perm = np.expm1(y_pred_log_perm)
                rmse_perm = _rmse(y_dollars, y_pred_perm)

                # Importance = how much RMSE got worse by permuting this feature
                importances[idx] += (rmse_perm - baseline_rmse)

                # Restore original column
                feature_df[col] = original_values

        # Average across repeats
        if n_repeats > 1:
            importances = importances / float(n_repeats)

        # Build result DataFrame
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "rmse_increase_dollars": importances,
        })
        # Add percent increase for business readability
        importance_df["percent_increase"] = (
            (importance_df["rmse_increase_dollars"] / baseline_rmse) * 100.0
        )
        importance_df["baseline_rmse_dollars"] = baseline_rmse
        importance_df = importance_df.sort_values("rmse_increase_dollars", ascending=False, ignore_index=True)

        # Persist to artifacts
        out_csv = os.path.join("artifacts", "feature_importance.csv")
        importance_df.to_csv(out_csv, index=False)

        # Return top-k as list of tuples
        top = importance_df.head(top_k)
        return list(zip(top["feature"].tolist(), top["rmse_increase_dollars"].tolist()))

    except Exception as e:
        raise CustomException(e)


if __name__ == "__main__":
    try:
        top_features = compute_top_feature_importances()
        print("Top 20 important features (by RMSE increase in $ when permuted):")
        for rank, (feat, score) in enumerate(top_features, start=1):
            print(f"{rank:2d}. {feat}: ${score:,.2f}")
        print("Saved full importances to artifacts/feature_importance.csv")
    except Exception as exc:
        print(f"Error computing feature importance: {exc}")

