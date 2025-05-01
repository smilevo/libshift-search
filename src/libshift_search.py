import pandas as pd
from pandas import DataFrame
import torch
import numpy as np
from collections import defaultdict

from src.search_util import SearchUtils


class LibshiftSearch:
    """
    Search for Top k replacement API using a Specific sentence transformer model, validated by
    the validation dataframe
    """

    def __init__(
        self,
        model_dict,
        removed_df: DataFrame,
        validation_df: DataFrame = None,
        snapshot_dictionary: dict = None,
        top_k: int = 10,
        features: list = None,
    ) -> None:
        """
        Args:
            model_name (str): Sentence transformer model name
            removed_df (DataFrame): Dataframe containing the removed libraries
                format: {id, name, args, library_name, path, code, docstring, feature_model_embedding}
            validation_df (DataFrame): Dataframe containing the validation libraries
                format : {Removed_Method, New_Method, removed_version, snapshot_versionm library_name, snapshot_id, removed_method_id}
            top_k (int): Top k libraries to be returned
        """

        self.removed_df = removed_df
        self.libraries = None
        self.validation_df = validation_df
        self.snapshot_dictionary = snapshot_dictionary
        self.top_k = top_k
        self.features = features
        self.removed_embeddings = defaultdict(dict)
        self.snapshot_embeddings = defaultdict(dict)
        self.libraries = None
        self.model_dict = {}
        self.__set_removed_df_lib()
        self.__infer_libraries()
        self.__set_models(model_dict)
        self.__set_embeddings()

    def __set_removed_df_lib(self):
        # from val_df set the removed_df library name
        # create a empty column in removed_df
        self.removed_df["library_name"] = None
        if self.validation_df is not None:
            # id of removed_df is same as removed_method_id in val_df
            for idx, row in self.validation_df.iterrows():
                removed_id = row["removed_method_id"]
                library_name = row["library_name"]
                # set the library name in removed_df
                self.removed_df.loc[
                    self.removed_df["id"] == removed_id, "library_name"
                ] = library_name

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
        similarity_data = []
        search_data = defaultdict(dict)
        for col, val in self.model_dict.items():
            for lib in self.libraries:
                search_metadata = {"model_name": val, "column": col, "lib": lib}
                val_df = self.validation_df[self.validation_df["library_name"] == lib]
                removed_embeddings = self.removed_embeddings[lib][col]
                snashot_embeddings = self.snapshot_embeddings[lib][col]
                su = SearchUtils(
                    removed_embeddings=removed_embeddings,
                    snapshot_embeddings=snashot_embeddings,
                    val_df=val_df,
                    mode=self.mode,
                    search_metadata=search_metadata,
                )
                similarity = su.search(k=self.top_k)
                similarity_data.extend(similarity)

        # go over every dict in similarity_data and make search data dict such that removed_id is the key and the value is a list of dicts
        for data in similarity_data:
            unique_id = data["unique_id"]
            if unique_id not in search_data.keys():
                search_data[unique_id] = []
            search_data[unique_id].append(data)
        return search_data

    def prepare_results(self, search_data: dict):
        """
        Prepare the results for the search experiment
        """
        match_data = []

        match_json = []
        model_types = ["name", "code", "docstring", "nodoc"]
        for id, data in search_data.items():
            entry = {
                "removed_id": data[0]["removed_id"],
                "library_name": data[0]["library_name"],
            }

            entry_json = {
                "removed_id": data[0]["removed_id"],
                "library_name": data[0]["library_name"],
                "removed_method_name": self.removed_df[
                    self.removed_df["id"] == data[0]["removed_id"]
                ]["name"].values[0],
            }

            for i, model_type in enumerate(model_types):
                model_data = data[i]
                entry[f"{model_type}_model"] = model_data["model_name"]
                entry[f"{model_type}_indices"] = model_data["top_indices"]
                entry[f"{model_type}_scores"] = model_data["top_scores"]
                entry[f"{model_type}_verified"] = model_data["verified"]

                entry_json["feature"] = model_type
                entry_json["model_name"] = model_data["model_name"]
                entry_json["top_indices"] = model_data["top_indices"]
                entry_json["top_scores"] = model_data["top_scores"]
                entry_json["has_match"] = model_data["verified"]
                entry_json["snapshot_ids"] = self._get_snapshot_id_from_indices(
                    model_data["top_indices"], model_data["library_name"]
                )
            match_data.append(entry)
            match_json.append(entry_json)

        # create dataframe from collected match data
        search_data = pd.DataFrame(match_data)

        # aggregate results
        agg_data = {
            f"{model_type}_model": search_data[f"{model_type}_model"].unique()[0]
            for model_type in model_types
        }
        agg_data.update(
            {
                f"total_{model_type}_hits": search_data[f"{model_type}_verified"].sum()
                for model_type in model_types
            }
        )
        agg_data["total_methods"] = len(search_data)

        agg_df = pd.DataFrame([agg_data])

        return search_data, agg_df, match_json

    def format_matches(self):
        pass

    def controller(self, mode: str = "cosine", topk: int = 10):
        self.top_k = topk
        self.mode = mode
        search_data = self.find_match()
        search_data, agg_data, match_json = self.prepare_results(search_data)

        return search_data, agg_data, match_json
