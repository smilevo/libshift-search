"""
Module for filtering model feature columns and reading repository snapshot data.

This code is provided strictly for research evaluation purposes only.
Redistribution, modification, or sharing of this code is prohibited.
All rights reserved by the author.

Author: anushkrishnav (GitHub)
Name: Anush Krishna V
Created: 1 May 2025
"""

import os
from typing import Dict, List, Optional
import pandas as pd


def filter_read_cols(models_dict: Dict[str, str]) -> List[str]:
    """
    Generate column names from a dictionary of model features and model names.

    Args:
        models_dict (Dict[str, str]): Dictionary with format {feature: model_name}.

    Returns:
        List[str]: List of formatted model column names.
    """
    model_columns = []
    for feature, model in models_dict.items():
        model_name = model.replace("/", "_")
        model_col_name = f"{feature}_{model_name}"
        model_columns.append(model_col_name)
    return model_columns


def get_snapshot_dict(
    folder: str, libs: List[str], filter_cols: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load parquet snapshots for each repository in `libs` from the given folder.

    Args:
        folder (str): Path to the directory containing parquet files.
        libs (List[str]): List of repository names.
        filter_cols (Optional[List[str]]): Columns to load from the parquet files.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary of DataFrames keyed by repository name.
    """
    snapshot_df: Dict[str, pd.DataFrame] = {}
    for repo_name in libs:
        if repo_name not in snapshot_df:
            file_path = os.path.join(folder, f"{repo_name}.parquet")
            df = (
                pd.read_parquet(file_path, columns=filter_cols)
                if filter_cols
                else pd.read_parquet(file_path)
            )
            snapshot_df[repo_name] = df
    return snapshot_df
