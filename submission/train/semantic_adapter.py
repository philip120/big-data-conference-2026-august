# train/semantic_adapter.py
"""
Semantic Adapter for MATLAB Code → Tree-Structured Semantic Operations

Extracts semantic operations from MATLAB code while PRESERVING TREE STRUCTURE.
Each operation includes its depth in the AST - this serves as position encoding.

Philosophy (like ViT but for code):
- ViT: patches have 2D position (row, col)
- This: operations have tree position (depth, and order)

Output format:
[
    {'depth': 0, 'type': 'function', 'text': 'function y = test(x)'},
    {'depth': 1, 'type': 'condition', 'text': 'x > 0'},
    {'depth': 2, 'type': 'assignment', 'text': 'y = x * 2'},
    ...
]
"""
import sys
import os
import re
from dataclasses import dataclass
from typing import Optional

# Add ANTLR parser to path. The generated matlabLexer/matlabParser may live
# either inside submission/ or one level up at the repo root.
_submission_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ANTLR_CANDIDATE_DIRS = [
    os.path.join(_submission_dir, "grammars-v4", "matlab"),
    os.path.join(os.path.dirname(_submission_dir), "grammars-v4", "matlab"),
]
for _d in _ANTLR_CANDIDATE_DIRS:
    if os.path.isfile(os.path.join(_d, "matlabParser.py")) and _d not in sys.path:
        sys.path.insert(0, _d)
        break

try:
    from antlr4 import InputStream, CommonTokenStream
    from matlabLexer import matlabLexer
    from matlabParser import matlabParser
    ANTLR_AVAILABLE = True
    print("[semantic_adapter] AST parser: ANTLR (grammars-v4/matlab)")
except ImportError:
    ANTLR_AVAILABLE = False
    print("[semantic_adapter] WARNING: ANTLR parser not found — using regex "
          "fallback. Tree structure will be approximate. Expected generated "
          "parser at one of: " + ", ".join(_ANTLR_CANDIDATE_DIRS))


@dataclass
class SemanticNode:
    """A semantic operation with tree position info."""
    depth: int          # Depth in AST (0 = root)
    type: str           # Operation type (assignment, condition, loop, etc.)
    text: str           # The actual code text
    parent_type: Optional[str] = None  # Parent node type for context

    def __repr__(self):
        indent = "  " * self.depth
        return f"{indent}[D{self.depth}] {self.type}: {self.text}"


# Semantic operation types we capture (smallest meaningful units)
CAPTURE_RULES = {
    'assignment_statement',
    'expression_statement',
    'global_statement',
    'clear_statement',
    'jump_statement',
}

# Structural rules we recurse into
STRUCTURAL_RULES = {
    'file_',
    'translation_unit',
    'statement_list',
    'selection_statement',
    'iteration_statement',
    'elseif_clause',
}


def get_node_text(node) -> str:
    """Extract full text from a parse tree node."""
    if node is None:
        return ""
    if not hasattr(node, 'children') or node.children is None:
        text = node.getText()
        return text if text else ""
    return "".join(get_node_text(child) for child in node.children)


def clean_text(text: str) -> str:
    """Clean extracted text."""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().rstrip(';').strip()
    # Remove newlines
    text = text.replace('\n', ' ').replace('\r', '')
    return text


def extract_tree_operations(tree, parser) -> list[SemanticNode]:
    """
    Extract semantic operations preserving tree depth.

    Returns list of SemanticNode with depth information.
    """
    operations = []

    def walk(node, depth: int = 0, parent_type: str = None):
        if node is None:
            return

        # Get rule name
        rule_name = None
        if hasattr(node, 'getRuleIndex'):
            rule_name = parser.ruleNames[node.getRuleIndex()]

        children = list(node.children) if hasattr(node, 'children') and node.children else []

        # === FUNCTION DECLARATION ===
        if rule_name == 'translation_unit':
            # Look for function keyword
            for i, child in enumerate(children):
                child_text = child.getText() if hasattr(child, 'getText') else ""
                if child_text == 'function':
                    # Get function declaration
                    if i + 1 < len(children):
                        func_decl = children[i + 1]
                        func_text = clean_text(get_node_text(func_decl))
                        operations.append(SemanticNode(
                            depth=depth,
                            type='function_decl',
                            text=f"function {func_text}",
                            parent_type=parent_type
                        ))
            # Recurse into statement list
            for child in children:
                if hasattr(child, 'getRuleIndex'):
                    child_rule = parser.ruleNames[child.getRuleIndex()]
                    if child_rule == 'statement_list':
                        walk(child, depth + 1, 'function')
            return

        # === SEMANTIC OPERATIONS (capture as single unit) ===
        if rule_name in CAPTURE_RULES:
            text = clean_text(get_node_text(node))
            if text and len(text) > 1:
                # Determine more specific type
                op_type = rule_name.replace('_statement', '')
                if '=' in text and op_type == 'expression':
                    op_type = 'assignment'
                operations.append(SemanticNode(
                    depth=depth,
                    type=op_type,
                    text=text,
                    parent_type=parent_type
                ))
            return  # Don't recurse into captured operations

        # === IF/ELSEIF/ELSE (selection_statement) ===
        if rule_name == 'selection_statement':
            i = 0
            while i < len(children):
                child = children[i]
                child_text = child.getText() if hasattr(child, 'getText') else ""

                if child_text == 'if':
                    # Next child is condition
                    if i + 1 < len(children):
                        cond = children[i + 1]
                        cond_text = clean_text(get_node_text(cond))
                        if cond_text:
                            operations.append(SemanticNode(
                                depth=depth,
                                type='if_condition',
                                text=f"if {cond_text}",
                                parent_type=parent_type
                            ))
                    # Find the statement_list (then body)
                    if i + 2 < len(children):
                        body = children[i + 2]
                        if hasattr(body, 'getRuleIndex'):
                            body_rule = parser.ruleNames[body.getRuleIndex()]
                            if body_rule == 'statement_list':
                                walk(body, depth + 1, 'if_then')
                    i += 3

                elif child_text == 'elseif':
                    if i + 1 < len(children):
                        cond = children[i + 1]
                        cond_text = clean_text(get_node_text(cond))
                        if cond_text:
                            operations.append(SemanticNode(
                                depth=depth,
                                type='elseif_condition',
                                text=f"elseif {cond_text}",
                                parent_type=parent_type
                            ))
                    if i + 2 < len(children):
                        body = children[i + 2]
                        if hasattr(body, 'getRuleIndex'):
                            walk(body, depth + 1, 'elseif')
                    i += 3

                elif child_text == 'else':
                    operations.append(SemanticNode(
                        depth=depth,
                        type='else',
                        text='else',
                        parent_type=parent_type
                    ))
                    # Next should be statement_list
                    if i + 1 < len(children):
                        body = children[i + 1]
                        if hasattr(body, 'getRuleIndex'):
                            walk(body, depth + 1, 'else')
                    i += 2

                elif child_text == 'end':
                    i += 1
                else:
                    # Check if it's a statement_list or elseif_clause
                    if hasattr(child, 'getRuleIndex'):
                        child_rule = parser.ruleNames[child.getRuleIndex()]
                        if child_rule == 'elseif_clause':
                            walk(child, depth, parent_type)
                    i += 1
            return

        # === FOR/WHILE LOOPS (iteration_statement) ===
        if rule_name == 'iteration_statement':
            loop_type = None
            loop_var = None
            loop_range = None

            for i, child in enumerate(children):
                child_text = child.getText() if hasattr(child, 'getText') else ""

                if child_text == 'for':
                    loop_type = 'for'
                elif child_text == 'while':
                    loop_type = 'while'
                elif child_text == '=' and loop_type == 'for':
                    # Previous child is variable, next is range
                    if i > 0:
                        loop_var = children[i - 1].getText()
                    if i + 1 < len(children):
                        loop_range = clean_text(get_node_text(children[i + 1]))
                elif hasattr(child, 'getRuleIndex'):
                    child_rule = parser.ruleNames[child.getRuleIndex()]
                    if child_rule == 'statement_list':
                        # First add the loop header
                        if loop_type == 'for' and loop_var and loop_range:
                            operations.append(SemanticNode(
                                depth=depth,
                                type='for_loop',
                                text=f"for {loop_var} = {loop_range}",
                                parent_type=parent_type
                            ))
                        elif loop_type == 'while':
                            # For while, the expression comes before statement_list
                            operations.append(SemanticNode(
                                depth=depth,
                                type='while_loop',
                                text=f"while {loop_range or '...'}",
                                parent_type=parent_type
                            ))
                        # Then recurse into body
                        walk(child, depth + 1, loop_type)
                    elif child_rule == 'expression' and loop_type == 'while':
                        loop_range = clean_text(get_node_text(child))
            return

        # === STATEMENT LIST (recurse) ===
        if rule_name == 'statement_list':
            for child in children:
                walk(child, depth, parent_type)
            return

        # === DEFAULT: recurse into children ===
        for child in children:
            walk(child, depth, parent_type)

    walk(tree)
    return operations


def code_to_nodes(code: str, as_objects: bool = False) -> list:
    """
    Convert MATLAB code to semantic operations with tree depth.

    Args:
        code: MATLAB source code
        as_objects: If True, return SemanticNode objects
                   If False, return strings with depth prefix

    Returns:
        List of operations (objects or strings depending on as_objects)
    """
    operations = []

    if ANTLR_AVAILABLE:
        try:
            input_stream = InputStream(code)
            lexer = matlabLexer(input_stream)
            stream = CommonTokenStream(lexer)
            parser_inst = matlabParser(stream)

            # Suppress errors
            from antlr4.error.ErrorListener import ErrorListener
            class Silent(ErrorListener):
                def syntaxError(self, *args, **kwargs):
                    pass
            lexer.removeErrorListeners()
            parser_inst.removeErrorListeners()
            lexer.addErrorListener(Silent())
            parser_inst.addErrorListener(Silent())

            tree = parser_inst.file_()
            operations = extract_tree_operations(tree, parser_inst)

        except Exception:
            pass

    # Fallback if ANTLR failed or not available
    if not operations:
        operations = fallback_extract(code)

    if as_objects:
        return operations
    else:
        # Return strings with depth info: "[D2] assignment: x = 1"
        return [f"[D{op.depth}] {op.type}: {op.text}" for op in operations]


def code_to_nodes_with_depth(code: str) -> tuple[list[str], list[int]]:
    """
    Convert code to (texts, depths) tuple.

    This is useful for the model:
    - texts: list of operation strings for CodeBERT
    - depths: list of depth values for position encoding

    Returns:
        (texts, depths) where both are aligned lists
    """
    operations = code_to_nodes(code, as_objects=True)
    texts = [op.text for op in operations]
    depths = [op.depth for op in operations]
    return texts, depths


def fallback_extract(code: str) -> list[SemanticNode]:
    """Fallback when ANTLR fails."""
    code = re.sub(r'%.*$', '', code, flags=re.MULTILINE)

    operations = []
    lines = code.split('\n')

    depth = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Track depth based on keywords
        if line.startswith(('function', 'if', 'for', 'while', 'switch', 'try')):
            op_type = line.split()[0] if line.split() else 'block'
            operations.append(SemanticNode(depth=depth, type=op_type, text=line.rstrip(';')))
            depth += 1
        elif line.startswith(('else', 'elseif', 'case', 'otherwise', 'catch')):
            op_type = line.split()[0]
            operations.append(SemanticNode(depth=max(0, depth-1), type=op_type, text=line.rstrip(';')))
        elif line == 'end':
            depth = max(0, depth - 1)
        elif '=' in line or line.endswith(';') or line.endswith(')'):
            # Statement
            operations.append(SemanticNode(depth=depth, type='statement', text=line.rstrip(';')))

    return operations


# Backwards compatibility
def code_to_ast_nodes(code: str) -> list[str]:
    return code_to_nodes(code, as_objects=False)


if __name__ == "__main__":
    test_code = """
    function y = test(x)
        if x > 0
            y = x * 2;
            disp('positive');
        elseif x < 0
            y = -x;
        else
            y = 0;
        end
        for i = 1:10
            disp(i);
        end
    end
    """

    print("=" * 70)
    print("TREE-STRUCTURED SEMANTIC OPERATIONS")
    print("=" * 70)

    ops = code_to_nodes(test_code, as_objects=True)
    print(f"\nFound {len(ops)} operations:\n")

    for op in ops:
        indent = "  " * op.depth
        print(f"{indent}[depth={op.depth}] {op.type}: {op.text}")

    print("\n" + "=" * 70)
    print("AS STRINGS (for model input)")
    print("=" * 70)

    strings = code_to_nodes(test_code, as_objects=False)
    for s in strings:
        print(f"  {s}")

    print("\n" + "=" * 70)
    print("SEPARATE TEXTS AND DEPTHS (for model)")
    print("=" * 70)

    texts, depths = code_to_nodes_with_depth(test_code)
    print(f"\nTexts ({len(texts)}):")
    for t in texts:
        print(f"  '{t}'")
    print(f"\nDepths: {depths}")

    print("\n" + "=" * 70)
    print("REAL FILE TEST")
    print("=" * 70)

    real_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "matlab_dataset/aludnam_MATLAB__analyzingtool_computebic/code.m"
    )
    try:
        with open(real_file) as f:
            real_code = f.read()
        ops = code_to_nodes(real_code, as_objects=True)
        print(f"\nFound {len(ops)} operations in bresenham.m:\n")
        for op in ops[:20]:
            indent = "  " * op.depth
            text = op.text[:50] + "..." if len(op.text) > 50 else op.text
            print(f"{indent}[D{op.depth}] {op.type}: {text}")
        if len(ops) > 20:
            print(f"\n  ... and {len(ops) - 20} more operations")
    except FileNotFoundError:
        print("  (file not found)")
