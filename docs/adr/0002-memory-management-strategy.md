# ADR-0002: Memory Management Strategy for Large Graphs

## Status
Accepted

## Context
Brain connectome graphs can contain 100,000+ nodes and billions of edges, which exceeds typical GPU memory limits (8-80GB). We need a strategy to handle these massive graphs while maintaining training efficiency and model performance.

Challenges:
- Full graphs don't fit in GPU memory
- Naive subsampling may lose important connectivity patterns
- Memory fragmentation during training
- Gradient computation for massive graphs

## Decision
We implement a **Hierarchical Sampling + Gradient Checkpointing** strategy:

1. **Hierarchical Graph Sampling**: Sample subgraphs based on brain anatomical hierarchy
2. **Gradient Checkpointing**: Trade computation for memory during backpropagation  
3. **Mixed Precision Training**: Use FP16 for most operations, FP32 for stability
4. **Lazy Loading**: Load graph components on-demand
5. **Intelligent Caching**: Cache frequently accessed subgraphs

## Consequences

### Positive
- Enables training on graphs that exceed GPU memory
- Preserves brain anatomical structure through hierarchical sampling
- Reduces memory usage by 50-70%
- Maintains model performance through intelligent sampling
- Supports scaling to larger datasets

### Negative
- Increased training time due to recomputation (gradient checkpointing)
- More complex data loading pipeline
- Requires careful tuning of sampling parameters
- Some loss of global connectivity information

## Implementation Details

### Hierarchical Sampling Strategy
```python
# Sample based on brain hierarchy:
# Level 1: Individual voxels/regions (local connectivity)
# Level 2: Brain networks (default mode, attention, etc.)
# Level 3: Hemispheric connections
# Level 4: Global brain structure
```

### Memory Budget Allocation
- 60%: Model parameters and activations
- 25%: Graph data (nodes, edges, features)
- 10%: Optimizer states
- 5%: Buffer for dynamic allocation

### Gradient Checkpointing Points
- After each GNN layer
- Before and after attention mechanisms
- At hierarchy level transitions

## Alternatives Considered

### Full Graph Processing
- **Pros:** No information loss, simpler implementation
- **Cons:** Impossible with current hardware constraints

### Random Subgraph Sampling
- **Pros:** Simple, well-established method
- **Cons:** Loses brain anatomical structure, may miss important patterns

### Graph Partitioning
- **Pros:** Maintains local structure
- **Cons:** Difficult to partition brain graphs meaningfully, edge cut problems

### CPU-GPU Hybrid Processing
- **Pros:** Unlimited memory capacity
- **Cons:** Severe performance bottleneck, complex implementation

## Related Decisions
- Builds on PyTorch Geometric choice (ADR-0001)
- Influences multi-modal fusion approach (ADR-0003)
- Impacts distributed training strategy