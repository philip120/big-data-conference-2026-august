from antlr4 import *
from matlabLexer import matlabLexer
from matlabParser import matlabParser

import sys

def print_tree(tree, parser, indent=0):
    """Pretty print the parse tree"""
    if tree is None:
        return

    # Get rule name or token text
    if hasattr(tree, 'getRuleIndex'):
        rule_name = parser.ruleNames[tree.getRuleIndex()]
        print("  " * indent + rule_name)
    else:
        text = tree.getText()
        if text.strip():
            print("  " * indent + repr(text))

    # Recurse into children
    if hasattr(tree, 'children') and tree.children:
        for child in tree.children:
            print_tree(child, parser, indent + 1)

if len(sys.argv) > 1:
    with open(sys.argv[1], 'r') as f:
        code = f.read()
else:
    code = "x = 1 + 2"

input_stream = InputStream(code)
lexer = matlabLexer(input_stream)
stream = CommonTokenStream(lexer)
parser = matlabParser(stream)
tree = parser.file_()

if len(sys.argv) > 2 and sys.argv[2] == '--flat':
    print(tree.toStringTree(recog=parser))
else:
    print_tree(tree, parser)
