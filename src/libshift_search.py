"""
Module for searching top-k API replacements using sentence-transformer embeddings
across removed and snapshot library versions. Supports multiple feature models.

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

from collections import defaultdict
import pandas as pd
from pandas import DataFrame
import torch
import numpy as np
from typing import List, Dict, Tuple, Any
from src.search_util import SearchUtils


class LibshiftSearch:
    """
    Search for Top k replacement API using a Specific sentence transformer model, validated by
    the validation dataframe
    """

    def __init__(
        self,
        model_dict: Any,
        removed_df: DataFrame,
        validation_df: DataFrame = None,
        snapshot_dictionary: Dict[str, DataFrame] = None,
        top_ks: list = [],
        features: List[str] = None,
        db_handler: Any = None,

    ) -> None:
        """
        Initialize the LibshiftSearch class.

        Args:
            model_dict (str | list | dict): Model name(s) or mapping per feature.
            removed_df (DataFrame): DataFrame of removed API methods.
            validation_df (DataFrame): Ground-truth mapping from removed to snapshot methods.
            snapshot_dictionary (dict): Dictionary mapping library name to snapshot DataFrame.
            top_k (int): Number of top matches to return.
            features (list): List of feature types to evaluate (e.g., ["name", "code"]).
        """

        self.removed_df = removed_df
        self.validation_df = validation_df
        self.snapshot_dictionary = snapshot_dictionary
        self.top_ks = top_ks
        self.features = features or []
        self.removed_embeddings = defaultdict(dict)
        self.snapshot_embeddings = defaultdict(dict)
        self.libraries = set()
        self.model_dict = {}
        self.db_handler = db_handler

        self.__set_removed_df_lib()
        self.__infer_libraries()
        self.__set_models(model_dict)
        self.__set_embeddings()



    def __set_removed_df_lib(self):
        """Assign library name in `removed_df` based on `validation_df`."""
        self.removed_df["library_name"] = None
        if self.validation_df is not None:
            # id of removed_df is same as removed_method_id in val_df
            for _, row in self.validation_df.iterrows():
                self.removed_df.loc[
                    self.removed_df["id"] == row["removed_method_id"], "library_name"
                ] = row["library_name"]


    def __set_models(self, models) -> bool:
        """
        Set the models for the removed and snapshot dataframes
        """
        if isinstance(models, str):
            for feature in self.features:
                self.model_dict[feature] = models
        elif isinstance(models, list):
            # check if feature and model list are same length
            if len(self.features) != len(models):
                raise ValueError("Features and models must be of same length")
            for feature, model in zip(self.features, models):
                self.model_dict[feature] = model
        elif isinstance(models, dict):
            # check if feature and model list are same length
            if len(self.features) != len(models.keys()):
                raise ValueError("Features and models must be of same length")
            for feature, model in zip(self.features, models.keys()):
                self.model_dict[feature] = models[model]
        else:
            raise ValueError("Model must be of type str, list or dict")

    def __infer_libraries(self) -> None:
        """
        Common libraries used by both removed, validation dataframes
        """
        if self.validation_df is not None:
            unique_val_list = self.validation_df["library_name"].unique().tolist()
            self.libraries = set(unique_val_list)

    def __set_embeddings(self) -> bool:
        """
        Set the embeddings for the removed and snapshot dataframes
        """
        for feature in self.features:
            model_name = self.model_dict[feature].replace("/", "_")
            model_col_name = f"{feature}_{model_name}"
            if model_col_name not in self.removed_df.columns:
                raise ValueError(
                    f"Feature model embedding column {model_col_name} not found in removed dataframe"
                )
            # group the removed_df on library_name
            removed_group = self.removed_df.groupby("library_name")
            for lib, group in removed_group:
                ids = group["id"].values
                self.removed_embeddings[lib][feature] = (
                    torch.tensor(np.vstack(group[f"{model_col_name}"].values)),
                    ids,
                    lib,
                )

            lib_remove = []
            for lib in self.libraries:
                if lib not in self.snapshot_dictionary.keys():
                    # remove the lib from the libraries
                    lib_remove.append(lib)
                    continue
                ids = self.snapshot_dictionary[lib]["id"].values
                if model_col_name not in self.snapshot_dictionary[lib].columns:
                    raise ValueError(
                        f"Feature model embedding column {model_col_name} not found in snapshot dataframe for library {lib}"
                    )
                self.snapshot_embeddings[lib][feature] = (
                    torch.tensor(
                        np.vstack(
                            self.snapshot_dictionary[lib][f"{model_col_name}"].values
                        )
                    ),
                    ids,
                    lib,
                )

            # remove the libs that are not in the snapshot dictionary
            for lib in lib_remove:
                self.libraries.remove(lib)
        return True

    def set_specific_libraries(self, libraries: list) -> bool:
        """
        Set specific libraries to be used for analysis
        """
        self.libraries = set(libraries)
        return True

    def _get_snapshot_id_from_indices(self, indices: list, library_name: str) -> str:
        """
        Get the id from the index
        Args:
            index (int): Index of the snapshot dataframe
            library_name (str): Library name
        Returns:
            str: ID of the snapshot dataframe
        """
        if library_name not in self.snapshot_dictionary.keys():
            raise ValueError(f"Library {library_name} not found in snapshot dictionary")
        # get the id from the index
        ids = self.snapshot_dictionary[library_name]["id"].values
        return ids[indices]

    def find_match(self):
        """
        Find the top k matches for each removed method in the validation dataframe
        Args:
            None

        Returns:
            search_data (dict): Dictionary of removed method id and the top k matches
        """
        similarity_data = []
        search_data = defaultdict(dict)
        
        for lib in self.libraries:
            val_df = self.validation_df[self.validation_df["library_name"] == lib]
            removed_embeddings = self.removed_embeddings[lib]
            snashot_embeddings = self.snapshot_embeddings[lib]
            su = SearchUtils(
                    removed_embeddings=removed_embeddings,
                    snapshot_embeddings=snashot_embeddings,
                    val_df=val_df,
                    mode=self.mode,
                    db_handler=self.db_handler,
                )
            for col, val in self.model_dict.items():        
                su.set_similarity_info({"model_name": val, "column": col, "lib": lib})
                similarity = su.search(topk=self.top_ks, column=col)
                similarity_data.extend(similarity)
            
            ## weighted search
            weighted_schema = {
                'name': 0.25,
                'nodoc': 0.75,
            }
            su.set_similarity_info({"model_name": "weighted", "column": "name_nodoc", "lib": lib})
            similarity = su.weighted_search(
                topk=self.top_ks,
                weighted_schema=weighted_schema,
            )
            similarity_data.extend(similarity)

        # go over every dict in similarity_data and make search data dict such that removed_id is the key and the value is a list of dicts
        for data in similarity_data:
            unique_id = data["unique_id"]
            if unique_id not in search_data.keys():
                search_data[unique_id] = []
            search_data[unique_id].append(data)
        return search_data

    def compute_combined_hits(self, match_df):
        """
        Compute the combined hits for the top-k matches
        Args:
            match_df (DataFrame): DataFrame of matches
        Returns:
            combined_hits (DataFrame): DataFrame of combined hits
        """

        combined_hits = []

        for k_val in self.top_ks:
            # Filter for current k
            df_k = match_df[match_df["k"] == k_val]

            # Consider only name, code, docstring model types
            df_k = df_k[df_k["model_type"].isin(["name", "code", "docstring"])]

            # Group by removed_id
            grouped = df_k.groupby("removed_id")

            correct_replacements = 0

            for _, group in grouped:
                if group["verified"].any():
                    correct_replacements += 1

            combined_hits.append({"Combined Top-k": k_val, "Correct Replacements": correct_replacements})
        
        return pd.DataFrame(combined_hits)

    def prepare_results(self, search_data: dict):
        """
        Prepare the results for the search experiment
        Args:
            search_data (dict): Dictionary of removed method id and the top k matches
        Returns:
            match_df (DataFrame): DataFrame of matches
            agg_df (DataFrame): DataFrame of aggregated results
            match_rows (list): List of dictionaries of matches
            combined_hits_df (DataFrame): DataFrame of combined hits
        """
        match_rows = []
        libwise_data = []
        topks = self.top_ks
        model_types = self.features

        for _, data_list in search_data.items():
            for model_data in data_list:
                match_rows.append({
                    "removed_id": model_data["removed_id"],
                    "library_name": model_data["library_name"],
                    "model_type": model_data["column"],
                    "model_name": model_data["model_name"],
                    "k": model_data["k"],
                    "verified": model_data["verified"],
                })

        match_df = pd.DataFrame(match_rows)

        # Initialize rows for each k
        agg_rows = []
        for k_val in topks:
            row = {"k": k_val, "total_methods": self.validation_df.shape[0]}

            # Get one model name per type (assuming consistent models per run)
            for model_type in model_types:
                filtered = match_df[(match_df["model_type"] == model_type) & (match_df["k"] == k_val)]
                model_name = filtered["model_name"].unique()
                row[f"{model_type}_model"] = model_name[0] if len(model_name) > 0 else None
                row[f"{model_type}_hits"] = filtered["verified"].sum()
            
            # For the weighted model
            filtered = match_df[(match_df["model_type"] == "name_nodoc") & (match_df["k"] == k_val)]
            row["wnodoc_hits"] = filtered["verified"].sum()

            agg_rows.append(row)
        
        columns = ['name_model', 'code_model', 'docstring_model','nodoc_model', 'k', 'name_hits', 'code_hits', 'docstring_hits', 'nodoc_hits', 'wnodoc_hits', 'total_methods']
        agg_df = pd.DataFrame(agg_rows, columns=columns)
        combined_hits_df = self.compute_combined_hits(match_df)

        return match_df, agg_df, match_rows, combined_hits_df  # match_rows acts as match_json


    def controller(self, mode: str = "cosine"):
        """
        Controller function to manage the search process and prepare results.

        Args:
            mode (str): The similarity mode to use for searching. Default is "cosine".
            Options include "cosine", "cosine-soft", "dot", "angular", "euclidean".
        Returns:
            search_data (dict): Dictionary of removed method id and the top k matches
            agg_df (DataFrame): DataFrame of aggregated results
            match_json (list): List of dictionaries of matches
            combined_hits_df (DataFrame): DataFrame of combined hits
        """
        self.mode = mode
        search_data = self.find_match()
        # dump search data to json
        text_data =  str(search_data)
        with open("output/search_data.txt", "w") as f:
            f.write(text_data)
        search_data, agg_data, match_json, combined_hits_df = self.prepare_results(search_data)

        return search_data, agg_data, match_json, combined_hits_df
