# shared/semantic_extractor.py
"""
Semantic Extraction

Extracts semantic operations from MATLAB code.
Each operation becomes a "pixel".
"""
import sys
import os

# Add train directory to path for semantic_adapter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.semantic_adapter import code_to_nodes


# Type vocabulary - maps operation types to embedding indices
TYPE_TO_ID = {
    'function': 0,
    'if': 1,
    'elseif': 2,
    'else': 3,
    'for': 4,
    'while': 5,
    'switch': 6,
    'case': 7,
    'otherwise': 8,
    'assignment': 9,
    'call': 10,
    'return': 11,
    'break': 12,
    'continue': 13,
    'try': 14,
    'catch': 15,
}
NUM_TYPES = len(TYPE_TO_ID)
MAX_DEPTH = 16
ID_TO_TYPE = {v: k for k, v in TYPE_TO_ID.items()}


class SemanticExtractor:
    """
    Extracts semantic operations from MATLAB code.

    Input:  MATLAB code string
    Output: dict with texts, depths, type_ids
    """

    def __call__(self, code: str) -> dict:
        """
        Extract semantic features from code.

        Args:
            code: MATLAB source code string

        Returns:
            dict with:
                - texts: list of operation text strings (for CodeBERT)
                - depths: list of depth integers (for depth embedding)
                - type_ids: list of type indices (for type embedding)
                - ops: list of SemanticNode objects (for Recursive Encoder)
        """
        ops = code_to_nodes(code, as_objects=True)

        if not ops:
            return {'texts': [], 'depths': [], 'type_ids': [], 'ops': []}

        texts = [op.text for op in ops]
        depths = [min(op.depth, MAX_DEPTH - 1) for op in ops]
        type_ids = [TYPE_TO_ID.get(op.type, 9) for op in ops]  # default: assignment

        return {
            'texts': texts,
            'depths': depths,
            'type_ids': type_ids,
            'ops': ops  # Add raw ops for recursive construction
        }


if __name__ == "__main__":
    # Test
    test_code = """
    function y = test(x)
        if x > 0
            y = x * 2;
        else
            y = 0;
        end
    end
    """

    extractor = SemanticExtractor()
    features = extractor(test_code)

    print("SemanticExtractor Test")
    print("=" * 50)
    print(f"Extracted {len(features['texts'])} pixels:\n")

    for i, (text, depth, tid) in enumerate(zip(
        features['texts'], features['depths'], features['type_ids']
    )):
        print(f"  [{i}] depth={depth}, type={ID_TO_TYPE[tid]:<10}, text=\"{text}\"")
