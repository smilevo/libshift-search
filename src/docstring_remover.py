"""
Module for removing docstrings from function definitions in Python AST.

This code is provided strictly for research evaluation purposes only.
Redistribution, modification, or sharing of this code is prohibited.
All rights reserved by the author.

Author: anushkrishnav (GitHub)
Name: Anush Krishna V
Created: 1 May 2025
"""

import ast
from typing import Any


class DocstringRemover(ast.NodeTransformer):
    """
    AST transformer that removes docstrings from function definitions.
    """

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        """
        Visit a function definition node and remove its docstring if present.

        Args:
            node (ast.FunctionDef): The function definition AST node.

        Returns:
            ast.AST: The modified function definition node.
        """
        self.generic_visit(node)

        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]

        return node
