"""
Module for removing docstrings from function definitions in Python AST.

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

Author: Anush Krishna V
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
