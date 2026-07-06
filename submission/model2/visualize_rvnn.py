import torch
import sys
import os

# Add root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model2.recursive_encoder import RecursiveEncoder
from model2.semantic_extractor import SemanticExtractorV2

# 1. Create Dummy Data
# Simulate projected pixel embeddings (1536 dim)
# Let's say we have 10 pixels in our code
embed_dim = 1536
num_pixels = 10
pixel_embeddings = torch.randn(1, num_pixels, embed_dim)

print(f"Dummy Pixel Embeddings: {pixel_embeddings.shape}\n")

# 2. Extract Tree Structure from Dummy Code
# We use a simple nested code example to trigger recursion
code = """
function [ qref ] = motionplan_with_rep( q0,qf,t1,t2,myrobot,obs,accur )
q = q0;
q_temp = q0;
while (norm(q_temp(end,1:5)-qf(end,1:5))>accur)
q_temp = q_temp + 0.01*att(q_temp,qf,myrobot) + 0.01*rep(q_temp,myrobot,obs);
q = vertcat(q,q_temp);
t = linspace(t1,t2,size(q,1));
qref = spline(t,q');
end
"""
print("--- Parsing Code Structure ---")
print(code)
print("------------------------------\n")

extractor = SemanticExtractorV2()
features = extractor(code)
roots = features['tree_roots']

print(f"Extracted {len(roots)} Tree Roots.")

# 3. Initialize RvNN
rvnn = RecursiveEncoder(embed_dim=embed_dim, max_branching=8)

# 4. Run Forward Pass with Debug
print("\n=== STARTING RECURSIVE VISUALIZATION ===\n")
global_vector = rvnn.forward_tree(roots, pixel_embeddings, debug=True)

print("\n=== FINAL OUTPUT ===")
print(f"Global Vector Shape: {global_vector.shape}")
