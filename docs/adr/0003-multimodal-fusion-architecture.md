# ADR-0003: Multi-Modal Data Fusion Architecture

## Status
Accepted

## Context
Brain connectivity analysis benefits from combining multiple data modalities:
- **Structural Connectivity**: White matter fiber tracts (DTI)
- **Functional Connectivity**: Synchronized brain activity (fMRI)
- **Effective Connectivity**: Causal relationships between regions

Each modality provides complementary information about brain organization. We need an architecture that effectively combines these modalities while handling their different characteristics:
- Different edge densities and sparsity patterns
- Different temporal dynamics
- Different noise characteristics
- Different spatial resolutions

## Decision
We implement an **Adaptive Late Fusion** architecture with modality-specific encoders:

1. **Modality-Specific Encoders**: Separate GNN branches for each modality
2. **Adaptive Fusion Layer**: Learned attention-based combination of modalities
3. **Shared Semantic Space**: Project all modalities to common representation
4. **Cross-Modal Attention**: Allow modalities to attend to each other
5. **Hierarchical Integration**: Fuse at multiple levels of brain organization

## Architecture Design

```python
class MultiModalBrainGNN(nn.Module):
    def __init__(self):
        # Modality-specific encoders
        self.structural_encoder = StructuralGNN()
        self.functional_encoder = FunctionalGNN() 
        self.effective_encoder = EffectiveGNN()
        
        # Fusion components
        self.cross_modal_attention = CrossModalAttention()
        self.adaptive_fusion = AdaptiveFusionLayer()
        self.shared_projection = SharedProjection()
```

## Consequences

### Positive
- Leverages complementary information from multiple modalities
- Adaptive fusion weights allow model to emphasize relevant modalities per task
- Modality-specific encoders can specialize for data characteristics
- Cross-modal attention captures inter-modality relationships
- Hierarchical fusion preserves multi-scale brain organization

### Negative
- Increased model complexity and parameter count
- Requires careful initialization and training procedures
- More complex data preprocessing pipeline
- Higher computational and memory requirements
- Risk of overfitting with limited multi-modal data

## Implementation Details

### Modality-Specific Encoders
- **Structural GNN**: Optimized for sparse, weighted graphs with geometric constraints
- **Functional GNN**: Handles dense, time-varying connectivity with temporal attention
- **Effective GNN**: Processes directed graphs with causal relationship modeling

### Fusion Strategy
1. **Early Fusion**: Concatenate raw modality features (baseline)
2. **Mid Fusion**: Combine intermediate representations (our choice)
3. **Late Fusion**: Fuse final embeddings (comparison method)

### Cross-Modal Attention Mechanism
```python
# Allow structural connectivity to inform functional processing
func_attended = cross_attention(
    query=functional_features,
    key=structural_features, 
    value=structural_features
)
```

## Alternatives Considered

### Early Fusion (Feature Concatenation)
- **Pros:** Simple, preserves all information
- **Cons:** Doesn't account for modality differences, may be dominated by high-variance modalities

### Ensemble Methods
- **Pros:** Simple to implement, good baseline performance  
- **Cons:** Doesn't capture cross-modal interactions, higher inference cost

### Graph Convolution on Multi-Layer Networks
- **Pros:** Principled approach for multi-layer graphs
- **Cons:** Limited flexibility, assumes similar processing for all modalities

### Modality Dropout Training
- **Pros:** Robust to missing modalities
- **Cons:** Doesn't fully leverage multi-modal information when available

## Validation Strategy

### Ablation Studies
- Single modality performance
- Different fusion strategies
- Cross-modal attention effectiveness
- Hierarchical vs. single-level fusion

### Multi-Modal Benchmarks
- Cognitive score prediction using multiple modalities
- Clinical diagnosis with structural + functional data
- Age prediction across modalities

## Related Decisions
- Builds on PyTorch Geometric framework (ADR-0001)
- Must work within memory constraints (ADR-0002)
- Influences visualization and interpretability tools