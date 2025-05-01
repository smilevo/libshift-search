from pandas import DataFrame
import ast
import astor
import re

from src.docstring_remover import DocstringRemover


class NoDocPreprocessor:
    """
    Preprocess the code to remove docstrings and comments
    to create the NoDoc Feature
    """

    def __init__(self, df: DataFrame):
        """
        Args:
            df (DataFrame): Dataframe containing the libraries
                format: {
                id,
                name,
                args,
                library_name,
                path,
                code,
                docstring,
                feature_model_embedding
            }
        """
        self.df = df
        self.ast_failed_indices = []
        self.regex_failed_indices = []

    def remove_docs(self, code):
        # Parse the code into an AST
        parsed = ast.parse(code)

        # Remove docstrings
        parsed = DocstringRemover().visit(parsed)
        ast.fix_missing_locations(parsed)

        # Convert AST back to source code
        clean_code = astor.to_source(parsed)

        # Remove comments manually (AST doesn't preserve them)
        clean_lines = []
        for line in clean_code.split("\n"):
            stripped = line.split("#")[0].rstrip()
            if stripped:
                clean_lines.append(stripped)

        return "\n".join(clean_lines)

    def regex_remove_docs(self, code):
        # Remove triple-quoted docstrings ("""...""" or '''...''')
        code = re.sub(r'("""|\'\'\')(.*?)(\1)', "", code, flags=re.DOTALL)
        # Remove inline comments
        code = re.sub(r"#.*", "", code)
        # Remove extra blank lines
        code = "\n".join([line.rstrip() for line in code.splitlines() if line.strip()])
        return code

    def safe_remove_docs(self, code, idx):
        try:
            return self.remove_docs(code)
        except Exception:
            self.ast_failed_indices.append(idx)
            try:
                return self.regex_remove_docs(code)
            except Exception:
                self.regex_failed_indices.append(idx)
                return code

    def create_nodoc_feature(self):
        """
        Create the NoDoc feature by removing docstrings
        and comments from the code
        """
        # check if code column exists
        if "code" not in self.df.columns:
            raise ValueError("DataFrame must contain a 'code' column")
        self.df["no_doc"] = [
            self.safe_remove_docs(code, idx) for idx, code in enumerate(self.df["code"])
        ]

        # check if no_doc column is created
        if "no_doc" not in self.df.columns:
            raise ValueError("NoDoc feature not created")
        # check if no_doc column is empty
        if self.df["no_doc"].isnull().all():
            raise ValueError("NoDoc feature is empty")
        return self.df
