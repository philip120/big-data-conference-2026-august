"""
Semantic Extractor for Model 2 (Tree-Aware)

This module extracts both sequential and tree-structured features.
Crucially, it converts the flat list of SemanticNodes into a proper nested tree
structure required for the Recursive Encoder.
"""
import sys
import os

# Add train directory to path for semantic_adapter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.semantic_adapter import code_to_nodes, SemanticNode

# Reuse type vocabulary from shared
from shared.semantic_extractor import TYPE_TO_ID, NUM_TYPES, MAX_DEPTH, ID_TO_TYPE

class TreeNode:
    """Wrapper for SemanticNode to support tree structure (children list)."""
    def __init__(self, semantic_node: SemanticNode, index: int):
        self.node = semantic_node
        self.index = index # Index in the flat pixel list (for gathering embeddings)
        self.children = [] # List of TreeNode

    def __repr__(self):
        return f"Tree(type={self.node.type}, idx={self.index}, children={len(self.children)})"

class SemanticExtractorV2:
    """
    Extracts tree-structured semantic operations.
    """

    def _build_tree(self, flat_ops: list[SemanticNode]) -> list[TreeNode]:
        """
        Reconstructs the AST hierarchy from the flat list of nodes based on depth.
        
        Logic:
        - Iterate through flat ops.
        - Maintain a stack of active parent nodes.
        - If current depth > stack top depth: add as child to stack top.
        - If current depth <= stack top depth: pop stack until finding correct parent.
        
        Returns:
            List of root TreeNodes.
        """
        if not flat_ops:
            return []

        # Convert all to TreeNodes
        tree_nodes = [TreeNode(op, i) for i, op in enumerate(flat_ops)] # i is the index of the operation in the flat list
        
        roots = []
        stack = [] # List of TreeNodes

        for node in tree_nodes:
            current_depth = node.node.depth
            
            # Pop stack until we find the parent (depth < current)
            while stack and stack[-1].node.depth >= current_depth:
                stack.pop()
            
            if stack:
                # Parent found
                stack[-1].children.append(node)
            else:
                # No parent -> Root node
                roots.append(node)
            
            # Push current node as potential parent for next nodes
            stack.append(node)
            
        return roots

    def __call__(self, code: str) -> dict:
        """
        Extract features.
        
        Returns:
            dict with:
            - standard features (texts, depths, type_ids)
            - 'tree_roots': List of root TreeNodes representing the full hierarchy.
        """
        ops = code_to_nodes(code, as_objects=True)

        if not ops:
            return {'texts': [], 'depths': [], 'type_ids': [], 'tree_roots': []}

        texts = [op.text for op in ops]
        depths = [min(op.depth, MAX_DEPTH - 1) for op in ops]
        type_ids = [TYPE_TO_ID.get(op.type, 9) for op in ops]

        # Build the tree structure
        tree_roots = self._build_tree(ops)

        return {
            'texts': texts,
            'depths': depths,
            'type_ids': type_ids,
            'tree_roots': tree_roots
        }

if __name__ == "__main__":
    test_code = """
    function y = test(x)
        if x > 0
            y = x * 2;
        else
            y = 0;
        end
    end
    """
    
    extractor = SemanticExtractorV2()
    features = extractor(test_code)
    
    print("SemanticExtractorV2 Test")
    print("=" * 50)
    
    def print_tree(node, level=0):
        indent = "  " * level
        print(f"{indent}- [{node.index}] {node.node.type} (children: {len(node.children)})")
        for child in node.children:
            print_tree(child, level + 1)
            
    print(f"Roots: {len(features['tree_roots'])}")
    for root in features['tree_roots']:
        print_tree(root)
