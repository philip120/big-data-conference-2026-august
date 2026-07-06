"""
Recursive Encoder (RvNN) for Model 2

Recursively aggregates tree-structured embeddings.
"""
import torch
import torch.nn as nn
from typing import List

class RecursiveEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1536,  # Input/Output dimension
        max_branching: int = 8, # Max children to aggregate
        hidden_dim: int = 3072, # Internal MLP dimension
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_branching = max_branching

        # Truncation telemetry: how often aggregate_children drops children
        # because a node has more than max_branching of them. Read via
        # truncation_stats() and reset via reset_truncation_stats().
        self.truncated_aggregations = 0
        self.total_aggregations = 0
        self.dropped_children = 0
        
        # Input dimension: Self Vector + Children Vectors
        # Children are flattened: max_branching * embed_dim
        # Parent self is added after MLP (residual) or concatenated?
        # Strategy:
        # 1. Aggregate children -> Child_Summary
        # 2. Combine Child_Summary + Parent_Self -> New_Parent
        
        # MLP to aggregate children.
        # No final LayerNorm: would pin output norm to sqrt(embed_dim)≈50,
        # causing the global_vector to dominate Qwen's residual stream.
        self.child_aggregator = nn.Sequential(
            nn.Linear(max_branching * embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Combiner: Parent + Child_Summary.
        self.combiner = nn.Linear(embed_dim * 2, embed_dim)
        
    def aggregate_children(self, child_vectors: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """
        Aggregate a batch of children groups.
        Args:
            child_vectors: [B, num_children, D]
        """
        B, N, D = child_vectors.shape
        device = child_vectors.device

        self.total_aggregations += 1
        if N > self.max_branching:
            self.truncated_aggregations += 1
            self.dropped_children += N - self.max_branching

        if debug:
            print(f"\n--- [RvNN Aggregation] ---")
            print(f"Input Children: {N} vectors")
            print(f"Shape: {child_vectors.shape}")
        
        # Pad to max_branching
        if N < self.max_branching:
            padding = torch.zeros(B, self.max_branching - N, D, device=device, dtype=child_vectors.dtype)
            padded = torch.cat([child_vectors, padding], dim=1)
            if debug:
                print(f"Padded with {self.max_branching - N} zero vectors.")
                print(f"Padded Shape: {padded.shape}")
        else:
            padded = child_vectors[:, :self.max_branching, :] # Truncate
            if debug:
                print(f"Truncated to {self.max_branching} vectors.")
            
        flattened = padded.reshape(B, -1)
        if debug:
            print(f"Flattened Input to MLP: {flattened.shape} (Size {flattened.numel()})")
            print("Encoding via MLP...")
            
        output = self.child_aggregator(flattened)
        
        if debug:
            print(f"MLP Output (Aggregated): {output.shape}")
            print("--------------------------\n")
            
        return output

    def truncation_stats(self) -> dict:
        """Fraction of aggregations that silently dropped children."""
        return {
            "max_branching": self.max_branching,
            "total_aggregations": self.total_aggregations,
            "truncated_aggregations": self.truncated_aggregations,
            "truncation_rate": (self.truncated_aggregations / self.total_aggregations
                                if self.total_aggregations else 0.0),
            "dropped_children": self.dropped_children,
        }

    def reset_truncation_stats(self):
        self.truncated_aggregations = 0
        self.total_aggregations = 0
        self.dropped_children = 0

    def forward_tree(self, nodes: List, pixel_embeddings: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """
        Recursively process a list of TreeNodes.
        
        Args:
            nodes: List of TreeNode (roots of current subtrees)
            pixel_embeddings: [1, Num_Pixels, D] - Lookup table for leaf vectors
                              (Assumes batch_size=1 for recursion simplicity)
            debug: If True, prints tensor shapes step-by-step.
                              
        Returns:
            Tensor [1, D] representing the aggregated vector of this forest.
            (Usually we call this on [root], so it returns the root vector).
        """
        if not nodes:
            return torch.zeros(1, self.embed_dim, device=pixel_embeddings.device)
            
        results = []
        for node in nodes:
            # Get clean text for visualization
            node_text = node.node.text.strip().replace('\n', ' ')
            if len(node_text) > 40:
                node_text = node_text[:37] + "..."
                
            if debug:
                print(f"\n[STEP] Processing Node [{node.index}] Type: '{node.node.type}'")
                print(f"       Code: \"{node_text}\"")
            
            # 1. Get Self Vector (Pixel embedding)
            # [1, D]
            self_vec = pixel_embeddings[:, node.index, :]
            
            # 2. Process Children Recursively
            if not node.children:
                # Leaf node: just return self (or transformed self)
                if debug:
                    print(f"       -> LEAF: Returning pixel embedding directly.")
                results.append(self_vec)
            else:
                # Recurse on children
                if debug:
                    print(f"       -> Has {len(node.children)} children. Recursing...")
                
                # This returns a list of [1, D] tensors, one per child
                child_vecs_list = [self.forward_tree([child], pixel_embeddings, debug) for child in node.children]
                
                # Stack: [1, Num_Children, D]
                # child_vecs_list elements are [1, D] (or [D]?) - forward_tree returns [1, D] currently if simple.
                # Let's verify return shape of forward_tree.
                # It returns output of aggregate_children -> [1, 1536]
                # OR self_vec -> [1, 1536]
                
                # So we want to stack N of [1, 1536] into [1, N, 1536]
                # cat(dim=0) -> [N, 1536]
                # unsqueeze(0) -> [1, N, 1536]
                
                if len(child_vecs_list) > 0:
                    child_vecs_tensor = torch.cat(child_vecs_list, dim=0).unsqueeze(0)
                else:
                    # Should not happen given 'if not node.children' check above
                    continue
                
                # Aggregate children: [1, D]
                if debug:
                    print(f"\n   [AGGREGATE] Aggregating children of Node [{node.index}] ('{node_text}'):")
                    for i, child in enumerate(node.children):
                        c_text = child.node.text.strip().replace('\n', ' ')
                        if len(c_text) > 30: c_text = c_text[:27] + "..."
                        print(f"       {i+1}. Child [{child.index}] ({child.node.type}): \"{c_text}\"")
                        
                child_summary = self.aggregate_children(child_vecs_tensor, debug)
                
                # Combine Self + Child Summary
                # [1, 2*D] -> [1, D]
                combined_input = torch.cat([self_vec, child_summary], dim=-1)
                combined = self.combiner(combined_input)
                
                if debug:
                    print(f"   [COMBINE] Concatenating Self (Node {node.index}) + Children Summary")
                    print(f"             Self Shape: {self_vec.shape} | Child Summary Shape: {child_summary.shape}")
                    print(f"             Result: {combined.shape} (New Parent Embedding)")
                
                results.append(combined)
        
        # If we were called on a list of siblings (roots), we usually return them as a tensor list
        # But if this function is "forward_tree" usually it processes one node.
        # However, to handle the top-level list of roots, we'll return the stack of results.
        # Wait, the signature says Tensor.
        # If called on multiple roots (forest), we should probably return them all?
        # But for "RecursiveEncoder", we usually want the Single Summary.
        
        # Let's adjust: This function processes a single NODE really well.
        # But if passed a list, it processes them in parallel?
        
        # If multiple roots, we assume they are siblings at the top level.
        # For the final output, we might want to aggregate them too?
        # Or usually the function has 1 root.
        
        if len(results) == 1:
            return results[0]
        else:
            # If multiple roots, return them stacked [1, N, D]? 
            # Or aggregate them one last time?
            # Let's aggregate them one last time to get a single Global Vector.
            # results elements are [1, 1536]
            # cat(dim=0) -> [N, 1536] -> unsqueeze(0) -> [1, N, 1536]
            stacked = torch.cat(results, dim=0).unsqueeze(0) # [1, N, D]
            if debug:
                print(f"Final Forest Aggregation of {len(results)} roots...")
            return self.aggregate_children(stacked, debug)
            
