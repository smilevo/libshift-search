import os
import pickle
import numpy as np
import pandas as pd
import torch
from sentence_transformers import util


class SearchUtils:
    def __init__(
        self,
        removed_embeddings: torch.Tensor = None,
        snapshot_embeddings: dict = None,
        val_df: pd.DataFrame = None,
        mode: str = "cosine",
        search_metadata: dict = None,
    ) -> None:
        self.removed_embeddings = removed_embeddings
        self.snapshot_embeddings = snapshot_embeddings
        self.val_df = val_df
        self.query_embedding = None
        self.target_embeddings = None
        self.mode = mode
        self.scores = None
        self.model_name = None
        self.column = None
        self.lib = None
        self.__set_similarity_info(search_metadata)

    def __set_similarity_info(self, data):
        """
        Set the similarity info
        Args:
            data (DataFrame): DataFrame containing the embeddings
        """
        self.model_name = data["model_name"]
        self.column = data["column"]
        self.lib = data["lib"]

    def __search_embeddings(
        self,
    ) -> list:
        """
        Search for the top k libraries using cosine similarity
        Returns:
            list: Similarity scores
        """

        # Calculate cosine similarity
        if self.mode == "cosine":
            similarities = util.cos_sim(self.query_embedding, self.target_embeddings)
        elif self.mode == "dot":
            similarities = torch.matmul(self.query_embedding, self.target_embeddings.T)
        elif self.mode == "euclidean":
            similarities = torch.cdist(
                self.query_embedding, self.target_embeddings, p=2
            )
        else:
            raise ValueError("Invalid mode. Choose from 'cosine', 'dot', 'euclidean'")
        # Convert to numpy array
        similarities = similarities.cpu().numpy()
        return similarities

    def __get_topk(self, k: int) -> list:
        """
        Get the top k methods
        Args:
            scores (np.ndarray): Similarity scores
            k (int): Top k methods to be returned
        Returns:
            list: Top k methods
        """
        # Get the top k methods
        if isinstance(self.scores, np.ndarray):
            # convert to tensor
            self.scores = torch.tensor(self.scores)
        top_results = torch.topk(self.scores, k=k, dim=1)
        top_indices = top_results.indices.flatten().tolist()
        top_scores = top_results.values.flatten().tolist()
        return top_indices, top_scores

    def store_similarity(
        self,
        data,
        col,
        k: int = 100,
    ) -> None:
        """
        Store the similarity scores as a parquet file
        Args:
            similarity (np.ndarray): Similarity scores
        """
        head_path = "data/similarity/"
        if not os.path.exists(head_path):
            os.makedirs(head_path, exist_ok=True)
        model_name = self.model_name.replace("/", "_")
        path = head_path + f"{model_name}_{self.lib}_{col}_{k}_{self.mode}_sim.pkl"
        # Check if the directory exists
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        # compress and store the dictionary
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(data, f)

    def search(self, k=10):
        """
        Search for the top k libraries using cosine similarity
        Returns:
            list: Similarity scores
        """
        # Calculate cosine similarity

        removed_idx_count = len(self.removed_embeddings[1])
        scores = []
        col = self.column
        model_name = self.model_name.replace("/", "_")
        path = f"../data/similarity/{model_name}_{self.lib}_{col}_{k}_{self.mode}_sim.pkl"
        # Check if the directory exists
        if os.path.exists(path):
            # read the similarity scores
            with open(path, "rb") as f:
                scores = pickle.load(f)
            return scores

        for i in range(removed_idx_count):
            lib = self.removed_embeddings[2]
            removed_id = self.removed_embeddings[1][i]
            self.query_embedding = self.removed_embeddings[0][i].unsqueeze(0)
            snapshot_ids = self.snapshot_embeddings[1]
            self.target_embeddings = self.snapshot_embeddings[0]
            self.scores = self.__search_embeddings()
            val_snapshot_ids = self.val_df[self.val_df['removed_method_id'] == removed_id]['snapshot_id'].values
            top_indices, top_scores = self.__get_topk(k=k)
            for val_snapshot_id in val_snapshot_ids:
                top_ids = [snapshot_ids[i] for i in top_indices]
                scores.append( {
                    "top_indices": top_indices,
                    "top_scores": top_scores,
                    "removed_id": removed_id,
                    "library_name": lib,
                    "target_id": val_snapshot_id,
                    "top_ids": top_ids,
                    "verified": val_snapshot_id in top_ids,
                    "column": col,
                    "model_name": model_name,
                    "unique_id" : f"{lib}_{removed_id}_{val_snapshot_id}",
                })
        self.store_similarity(scores, col, k=k)
        return scores