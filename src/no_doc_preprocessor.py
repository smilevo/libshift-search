"""
Module for removing docstrings and comments from Python source code
to generate a 'NoDoc' version of the code for model evaluation.

This code is provided strictly for research evaluation purposes only.
Redistribution, modification, or sharing of this code is prohibited.
All rights reserved by the author.

Author: anushkrishnav (GitHub)
Name: Anush Krishna V
Created: 1 May 2025
"""

import ast
import re
import astor
from typing import List
from pandas import DataFrame
from src.docstring_remover import DocstringRemover


class NoDocPreprocessor:
    """
    Preprocessor to remove docstrings and comments from Python code.

    Designed to create a 'no_doc' version of source code for evaluating
    code models without documentation context.
    """

    def __init__(self, df: DataFrame):
        """
        Initialize the NoDocPreprocessor.

        Args:
            df (DataFrame): DataFrame with at least a 'code' column.
                Expected columns include:
                ['id', 'name', 'args', 'library_name', 'path', 'code', 'docstring', 'feature_model_embedding']
        """
        self.df = df
        self.ast_failed_indices: List[int] = []
        self.regex_failed_indices: List[int] = []

    def remove_docs(self, code: str) -> str:
        """
        Remove docstrings using AST and comments using regex.

        Args:
            code (str): Python source code as a string.

        Returns:
            str: Cleaned source code without docstrings or comments.
        """
        parsed = ast.parse(code)
        parsed = DocstringRemover().visit(parsed)
        ast.fix_missing_locations(parsed)
        clean_code = astor.to_source(parsed)

        # Remove inline comments
        clean_lines = []
        for line in clean_code.split("\n"):
            stripped = line.split("#")[0].rstrip()
            if stripped:
                clean_lines.append(stripped)

        return "\n".join(clean_lines)

    def regex_remove_docs(self, code: str) -> str:
        """
        Fallback method to remove docstrings and comments using regex.

        Args:
            code (str): Python source code as a string.

        Returns:
            str: Cleaned source code.
        """
        code = re.sub(r'("""|\'\'\')(.*?)(\1)', "", code, flags=re.DOTALL)
        code = re.sub(r"#.*", "", code)
        code = "\n".join([line.rstrip() for line in code.splitlines() if line.strip()])
        return code

    def safe_remove_docs(self, code: str, idx: int) -> str:
        """
        Attempt to clean code using AST first, then fallback to regex if needed.

        Args:
            code (str): Python source code as a string.
            idx (int): Index of the row in the dataframe for error tracking.

        Returns:
            str: Cleaned code, or original code if both methods fail.
        """
        try:
            return self.remove_docs(code)
        except Exception:
            self.ast_failed_indices.append(idx)
            try:
                return self.regex_remove_docs(code)
            except Exception:
                self.regex_failed_indices.append(idx)
                return code

    def create_nodoc_feature(self) -> DataFrame:
        """
        Generate a new column 'no_doc' in the dataframe by stripping docstrings and comments.

        Returns:
            DataFrame: DataFrame with an additional 'no_doc' column.
        """
        if "code" not in self.df.columns:
            raise ValueError("DataFrame must contain a 'code' column.")

        self.df["no_doc"] = [
            self.safe_remove_docs(code, idx) for idx, code in enumerate(self.df["code"])
        ]

        if self.df["no_doc"].isnull().all():
            raise ValueError("NoDoc feature is empty after processing.")

        return self.df
