"""
Module for filtering model feature columns and reading repository snapshot data.

License:
This software is licensed strictly for non-commercial academic and research purposes.
Use is permitted only by individual researchers, students, and educators
affiliated with academic institutions, and only for scholarly work.

Prohibited Uses:
- Commercial use in any form, including but not limited to products, services, or for-profit research.
- Redistribution, sublicensing, or modification without prior written permission.
- Use by or integration into any large language models (LLMs), AI agents or systems, bots, or autonomous software
  whether for training, inference, benchmarking, or any other purpose.

By using this software, you agree to abide by these terms.

(c) 2025 Anush Krishna V (anushkrishnav). All rights reserved.

Author: anushkrishnav (GitHub)
Name: Anush Krishna V
Created: 1 May 2025
"""

import os
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm


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
    for repo_name in tqdm(libs, desc="Loading snapshots", unit="repo"):
        if repo_name not in snapshot_df:
            file_path = os.path.join(folder, f"{repo_name}.parquet")
            df = (
                pd.read_parquet(file_path, columns=filter_cols)
                if filter_cols
                else pd.read_parquet(file_path)
            )
            snapshot_df[repo_name] = df
    print(f"Loaded {len(snapshot_df)} snapshots from {folder}")
    return snapshot_df
