import pandas as pd
import os


def filter_read_cols(
    models_dict: dict,
):
    """
    Filter the models dictionary to only include the columns that are present in the removed dataframe
    Args:
        models_dict (dict): Dictionary containing the models
            format: {name: model_name, code: model_name, docstring: model_name, nodoc: model_name}
    Returns:
        list: List of model_columns
    """
    model_columns = []
    for feature, model in models_dict.items():
        model_name = model.replace("/", "_")
        model_col_name = f"{feature}_{model_name}"
        model_columns.append(model_col_name)
    return model_columns


def get_snapshot_dict(folder: str, libs: list, filter_cols: list = None):
    snapshot_df = {}
    for repo_selected in libs:
        if repo_selected not in snapshot_df.keys():
            tsnapshot_path = os.path.join(folder, f"{repo_selected}.parquet")
            if filter_cols:
                tsnapshot_path = pd.read_parquet(tsnapshot_path, columns=filter_cols)
            else:
                tsnapshot_path = pd.read_parquet(tsnapshot_path)
            snapshot_df[repo_selected] = tsnapshot_path
    return snapshot_df
