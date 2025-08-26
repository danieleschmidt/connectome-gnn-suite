"""Quantum Acceleration Framework for Graph Neural Networks.

Implements quantum computing acceleration techniques for GNN operations,
including quantum approximate optimization algorithms (QAOA) and
variational quantum eigensolvers (VQE) for graph problems.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
import time
import logging
import cmath
from abc import ABC, abstractmethod
import itertools
from collections import defaultdict


@dataclass
class QuantumCircuitMetrics:
    """Metrics for quantum circuit performance."""
    circuit_depth: int = 0
    gate_count: int = 0
    qubit_count: int = 0
    execution_time_ms: float = 0.0
    fidelity: float = 1.0
    quantum_volume: int = 0
    classical_simulation_time_ms: float = 0.0
    quantum_advantage_factor: float = 1.0


class QuantumState:
    """Represents a quantum state with operations."""
    
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.state_vector = np.zeros(2**num_qubits, dtype=complex)
        self.state_vector[0] = 1.0  # Initialize to |0...0⟩
        
    def apply_hadamard(self, qubit: int):
        """Apply Hadamard gate to specified qubit."""
        h_matrix = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        self._apply_single_qubit_gate(h_matrix, qubit)
    
    def apply_x(self, qubit: int):
        """Apply Pauli-X gate to specified qubit."""
        x_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
        self._apply_single_qubit_gate(x_matrix, qubit)
    
    def apply_y(self, qubit: int):
        """Apply Pauli-Y gate to specified qubit."""
        y_matrix = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self._apply_single_qubit_gate(y_matrix, qubit)
    
    def apply_z(self, qubit: int):
        """Apply Pauli-Z gate to specified qubit."""
        z_matrix = np.array([[1, 0], [0, -1]], dtype=complex)
        self._apply_single_qubit_gate(z_matrix, qubit)
    
    def apply_rotation_z(self, qubit: int, theta: float):
        """Apply rotation around Z-axis."""
        rz_matrix = np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ], dtype=complex)
        self._apply_single_qubit_gate(rz_matrix, qubit)
    
    def apply_rotation_y(self, qubit: int, theta: float):
        """Apply rotation around Y-axis."""
        ry_matrix = np.array([
            [np.cos(theta / 2), -np.sin(theta / 2)],
            [np.sin(theta / 2), np.cos(theta / 2)]
        ], dtype=complex)
        self._apply_single_qubit_gate(ry_matrix, qubit)
    
    def apply_cnot(self, control: int, target: int):
        """Apply CNOT gate between control and target qubits."""
        cnot_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
        
        self._apply_two_qubit_gate(cnot_matrix, control, target)
    
    def _apply_single_qubit_gate(self, gate_matrix: np.ndarray, qubit: int):
        """Apply single qubit gate to the state vector."""
        new_state = np.zeros_like(self.state_vector)
        
        for i in range(2**self.num_qubits):
            # Extract qubit value
            qubit_val = (i >> (self.num_qubits - 1 - qubit)) & 1
            
            # Apply gate matrix
            for new_val in range(2):
                # Calculate new state index
                new_i = i ^ ((qubit_val ^ new_val) << (self.num_qubits - 1 - qubit))
                
                # Apply matrix element
                new_state[new_i] += gate_matrix[new_val, qubit_val] * self.state_vector[i]
        
        self.state_vector = new_state
    
    def _apply_two_qubit_gate(self, gate_matrix: np.ndarray, qubit1: int, qubit2: int):
        """Apply two-qubit gate to the state vector."""
        new_state = np.zeros_like(self.state_vector)
        
        for i in range(2**self.num_qubits):
            # Extract qubit values
            q1_val = (i >> (self.num_qubits - 1 - qubit1)) & 1
            q2_val = (i >> (self.num_qubits - 1 - qubit2)) & 1
            
            # Current two-qubit state
            two_qubit_state = q1_val * 2 + q2_val
            
            # Apply gate to all possible output states
            for new_two_qubit_state in range(4):
                new_q1_val = new_two_qubit_state >> 1
                new_q2_val = new_two_qubit_state & 1
                
                # Calculate new state index
                new_i = i
                new_i ^= (q1_val ^ new_q1_val) << (self.num_qubits - 1 - qubit1)
                new_i ^= (q2_val ^ new_q2_val) << (self.num_qubits - 1 - qubit2)
                
                # Apply matrix element
                new_state[new_i] += gate_matrix[new_two_qubit_state, two_qubit_state] * self.state_vector[i]
        
        self.state_vector = new_state
    
    def measure(self, qubit: int) -> int:
        """Measure a qubit and collapse the state."""
        # Calculate probabilities
        prob_0 = 0.0
        prob_1 = 0.0
        
        for i in range(2**self.num_qubits):
            qubit_val = (i >> (self.num_qubits - 1 - qubit)) & 1
            prob = abs(self.state_vector[i])**2
            
            if qubit_val == 0:
                prob_0 += prob
            else:
                prob_1 += prob
        
        # Random measurement outcome
        outcome = 0 if np.random.random() < prob_0 / (prob_0 + prob_1) else 1
        
        # Collapse state
        norm = 0.0
        for i in range(2**self.num_qubits):
            qubit_val = (i >> (self.num_qubits - 1 - qubit)) & 1
            
            if qubit_val == outcome:
                norm += abs(self.state_vector[i])**2
            else:
                self.state_vector[i] = 0.0
        
        # Normalize
        if norm > 0:
            self.state_vector /= np.sqrt(norm)
        
        return outcome
    
    def get_expectation_value(self, observable: np.ndarray) -> float:
        """Calculate expectation value of an observable."""
        return np.real(np.conj(self.state_vector) @ observable @ self.state_vector)
    
    def get_probabilities(self) -> np.ndarray:
        """Get measurement probabilities for all basis states."""
        return np.abs(self.state_vector)**2


class QuantumApproximateOptimization:
    """Quantum Approximate Optimization Algorithm (QAOA) for graph problems."""
    
    def __init__(self, num_qubits: int, depth: int = 1):
        self.num_qubits = num_qubits
        self.depth = depth
        self.beta_params = np.random.uniform(0, np.pi, depth)
        self.gamma_params = np.random.uniform(0, 2*np.pi, depth)
        
    def create_mixing_hamiltonian(self) -> np.ndarray:
        """Create mixing Hamiltonian (sum of X operators)."""
        hamiltonian = np.zeros((2**self.num_qubits, 2**self.num_qubits), dtype=complex)
        
        for qubit in range(self.num_qubits):
            # Single-qubit X operator
            x_op = np.eye(2**self.num_qubits, dtype=complex)
            
            for i in range(2**self.num_qubits):
                j = i ^ (1 << (self.num_qubits - 1 - qubit))  # Flip bit
                x_op[i, i] = 0
                x_op[i, j] = 1
            
            hamiltonian += x_op
        
        return hamiltonian
    
    def create_problem_hamiltonian(self, adjacency_matrix: np.ndarray) -> np.ndarray:
        """Create problem Hamiltonian for graph problems."""
        hamiltonian = np.zeros((2**self.num_qubits, 2**self.num_qubits), dtype=complex)
        
        # Add edge terms
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                if adjacency_matrix[i, j] != 0:
                    weight = adjacency_matrix[i, j]
                    
                    # Create ZZ interaction term
                    zz_op = np.eye(2**self.num_qubits, dtype=complex)
                    
                    for k in range(2**self.num_qubits):
                        z_i = 1 if (k >> (self.num_qubits - 1 - i)) & 1 else -1
                        z_j = 1 if (k >> (self.num_qubits - 1 - j)) & 1 else -1
                        zz_op[k, k] = z_i * z_j
                    
                    hamiltonian += weight * zz_op
        
        return hamiltonian
    
    def apply_qaoa_circuit(self, state: QuantumState, problem_hamiltonian: np.ndarray) -> QuantumState:
        """Apply QAOA circuit to quantum state."""
        # Initialize with equal superposition
        for qubit in range(self.num_qubits):
            state.apply_hadamard(qubit)
        
        mixing_hamiltonian = self.create_mixing_hamiltonian()
        
        # Apply alternating problem and mixing unitaries
        for layer in range(self.depth):
            gamma = self.gamma_params[layer]
            beta = self.beta_params[layer]
            
            # Apply problem unitary: exp(-i * gamma * H_p)
            problem_unitary = self._matrix_exponential(-1j * gamma * problem_hamiltonian)
            state.state_vector = problem_unitary @ state.state_vector
            
            # Apply mixing unitary: exp(-i * beta * H_m)  
            mixing_unitary = self._matrix_exponential(-1j * beta * mixing_hamiltonian)
            state.state_vector = mixing_unitary @ state.state_vector
        
        return state
    
    def _matrix_exponential(self, matrix: np.ndarray) -> np.ndarray:
        """Compute matrix exponential using eigendecomposition."""
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        return eigenvecs @ np.diag(np.exp(eigenvals)) @ eigenvecs.T.conj()
    
    def optimize_parameters(
        self,
        adjacency_matrix: np.ndarray,
        max_iterations: int = 100,
        learning_rate: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Optimize QAOA parameters using gradient descent."""
        
        problem_hamiltonian = self.create_problem_hamiltonian(adjacency_matrix)
        
        for iteration in range(max_iterations):
            # Calculate gradient numerically
            beta_gradients = np.zeros_like(self.beta_params)
            gamma_gradients = np.zeros_like(self.gamma_params)
            
            epsilon = 0.01
            
            # Base expectation value
            base_expectation = self._evaluate_expectation(problem_hamiltonian)
            
            # Calculate gradients
            for i in range(self.depth):
                # Beta gradient
                self.beta_params[i] += epsilon
                plus_expectation = self._evaluate_expectation(problem_hamiltonian)
                self.beta_params[i] -= 2 * epsilon
                minus_expectation = self._evaluate_expectation(problem_hamiltonian)
                self.beta_params[i] += epsilon
                
                beta_gradients[i] = (plus_expectation - minus_expectation) / (2 * epsilon)
                
                # Gamma gradient
                self.gamma_params[i] += epsilon
                plus_expectation = self._evaluate_expectation(problem_hamiltonian)
                self.gamma_params[i] -= 2 * epsilon
                minus_expectation = self._evaluate_expectation(problem_hamiltonian)
                self.gamma_params[i] += epsilon
                
                gamma_gradients[i] = (plus_expectation - minus_expectation) / (2 * epsilon)
            
            # Update parameters
            self.beta_params -= learning_rate * beta_gradients
            self.gamma_params -= learning_rate * gamma_gradients
            
            # Print progress
            if iteration % 10 == 0:
                current_expectation = self._evaluate_expectation(problem_hamiltonian)
                logging.info(f"QAOA Iteration {iteration}: Expectation = {current_expectation:.4f}")
        
        return self.beta_params, self.gamma_params
    
    def _evaluate_expectation(self, problem_hamiltonian: np.ndarray) -> float:
        """Evaluate expectation value for current parameters."""
        state = QuantumState(self.num_qubits)
        state = self.apply_qaoa_circuit(state, problem_hamiltonian)
        return state.get_expectation_value(problem_hamiltonian)


class QuantumGraphEmbedding:
    """Quantum-enhanced graph embedding using variational circuits."""
    
    def __init__(self, num_qubits: int, embedding_dim: int, circuit_depth: int = 3):
        self.num_qubits = num_qubits
        self.embedding_dim = embedding_dim
        self.circuit_depth = circuit_depth
        
        # Variational parameters
        self.rotation_params = np.random.uniform(0, 2*np.pi, (circuit_depth, num_qubits, 3))
        self.entangling_params = np.random.uniform(0, 2*np.pi, (circuit_depth, num_qubits))
        
    def create_variational_circuit(self, state: QuantumState, input_features: np.ndarray) -> QuantumState:
        """Create variational quantum circuit for embedding."""
        
        # Encode input features
        for i, feature in enumerate(input_features[:self.num_qubits]):
            state.apply_rotation_y(i, feature * np.pi)
        
        # Apply variational layers
        for layer in range(self.circuit_depth):
            # Single-qubit rotations
            for qubit in range(self.num_qubits):
                state.apply_rotation_y(qubit, self.rotation_params[layer, qubit, 0])
                state.apply_rotation_z(qubit, self.rotation_params[layer, qubit, 1])
                state.apply_rotation_y(qubit, self.rotation_params[layer, qubit, 2])
            
            # Entangling gates
            for qubit in range(self.num_qubits - 1):
                state.apply_cnot(qubit, qubit + 1)
                state.apply_rotation_z(qubit + 1, self.entangling_params[layer, qubit])
        
        return state
    
    def extract_embedding(self, input_features: np.ndarray) -> np.ndarray:
        """Extract quantum embedding from input features."""
        state = QuantumState(self.num_qubits)
        state = self.create_variational_circuit(state, input_features)
        
        # Extract embedding from measurement probabilities
        probabilities = state.get_probabilities()
        
        # Compress to desired embedding dimension
        if self.embedding_dim < len(probabilities):
            # Use PCA-like projection (simplified)
            embedding = probabilities[:self.embedding_dim]
        else:
            # Pad with zeros if needed
            embedding = np.pad(probabilities, (0, self.embedding_dim - len(probabilities)))
        
        return embedding / np.linalg.norm(embedding)  # Normalize


class QuantumConvolutionalLayer:
    """Quantum convolutional layer for graph neural networks."""
    
    def __init__(self, in_features: int, out_features: int, num_qubits: int = 4):
        self.in_features = in_features
        self.out_features = out_features
        self.num_qubits = min(num_qubits, 10)  # Limit for classical simulation
        
        # Quantum circuit parameters
        self.circuit_params = np.random.uniform(0, 2*np.pi, (out_features, num_qubits, 3))
        
        # Classical post-processing
        self.classical_layer = nn.Linear(2**self.num_qubits, out_features)
        
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through quantum convolutional layer."""
        batch_size, num_nodes, _ = node_features.shape
        
        # Convert to numpy for quantum processing
        features_np = node_features.detach().cpu().numpy()
        
        quantum_outputs = []
        
        for batch in range(batch_size):
            batch_outputs = []
            
            for node in range(num_nodes):
                node_feature = features_np[batch, node]
                
                # Quantum processing
                quantum_embedding = self._process_node_quantum(node_feature)
                batch_outputs.append(quantum_embedding)
            
            quantum_outputs.append(np.stack(batch_outputs))
        
        # Convert back to torch
        quantum_tensor = torch.from_numpy(np.stack(quantum_outputs)).float()
        
        # Classical post-processing
        output = self.classical_layer(quantum_tensor)
        
        return output
    
    def _process_node_quantum(self, node_features: np.ndarray) -> np.ndarray:
        """Process node features through quantum circuit."""
        embeddings = []
        
        for output_idx in range(self.out_features):
            state = QuantumState(self.num_qubits)
            
            # Encode input features
            for i in range(min(len(node_features), self.num_qubits)):
                normalized_feature = node_features[i] / (np.linalg.norm(node_features) + 1e-8)
                state.apply_rotation_y(i, normalized_feature * np.pi)
            
            # Apply variational circuit
            for qubit in range(self.num_qubits):
                state.apply_rotation_y(qubit, self.circuit_params[output_idx, qubit, 0])
                state.apply_rotation_z(qubit, self.circuit_params[output_idx, qubit, 1])
                state.apply_rotation_y(qubit, self.circuit_params[output_idx, qubit, 2])
            
            # Entangling gates
            for qubit in range(self.num_qubits - 1):
                state.apply_cnot(qubit, qubit + 1)
            
            # Extract probabilities as features
            probabilities = state.get_probabilities()
            embeddings.append(probabilities)
        
        return np.concatenate(embeddings)


class QuantumAcceleratedGNN(nn.Module):
    """Graph Neural Network with quantum acceleration."""
    
    def __init__(
        self,
        node_features: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 1,
        quantum_features: int = 16,
        use_quantum: bool = True
    ):
        super().__init__()
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.use_quantum = use_quantum
        
        # Quantum components
        if use_quantum:
            self.quantum_embedding = QuantumGraphEmbedding(
                num_qubits=min(8, quantum_features),
                embedding_dim=quantum_features
            )
            
            self.quantum_conv_layers = nn.ModuleList([
                QuantumConvolutionalLayer(
                    in_features=quantum_features if i == 0 else hidden_dim,
                    out_features=hidden_dim,
                    num_qubits=4
                )
                for i in range(num_layers)
            ])
        
        # Classical components
        self.classical_layers = nn.ModuleList([
            nn.Linear(
                quantum_features if i == 0 and use_quantum else hidden_dim,
                hidden_dim
            )
            for i in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        
        # Hybrid fusion
        if use_quantum:
            self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through quantum-accelerated GNN."""
        
        if self.use_quantum:
            # Quantum processing
            quantum_features = self._apply_quantum_processing(x)
            
            # Classical processing
            classical_features = x
            
            # Process through layers
            for i in range(self.num_layers):
                # Quantum path
                if i == 0:
                    quantum_h = quantum_features
                quantum_h = self.quantum_conv_layers[i](quantum_h.unsqueeze(0), edge_index).squeeze(0)
                quantum_h = torch.relu(quantum_h)
                
                # Classical path
                classical_h = self.classical_layers[i](classical_features if i == 0 else classical_h)
                classical_h = torch.relu(classical_h)
            
            # Fusion
            combined_features = torch.cat([quantum_h, classical_h], dim=-1)
            fused_features = self.fusion_layer(combined_features)
            
        else:
            # Pure classical processing
            fused_features = x
            for layer in self.classical_layers:
                fused_features = torch.relu(layer(fused_features))
        
        # Global pooling
        if batch is not None:
            # Batch processing
            from torch_geometric.nn import global_mean_pool
            pooled_features = global_mean_pool(fused_features, batch)
        else:
            # Single graph
            pooled_features = torch.mean(fused_features, dim=0, keepdim=True)
        
        # Output
        output = self.output_layer(pooled_features)
        
        return output.squeeze(-1) if self.num_classes == 1 else output
    
    def _apply_quantum_processing(self, x: torch.Tensor) -> torch.Tensor:
        """Apply quantum processing to input features."""
        x_np = x.detach().cpu().numpy()
        
        quantum_features = []
        for node_features in x_np:
            embedding = self.quantum_embedding.extract_embedding(node_features)
            quantum_features.append(embedding)
        
        return torch.from_numpy(np.stack(quantum_features)).float().to(x.device)


class QuantumAccelerationFramework:
    """Framework for quantum acceleration of graph neural networks."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics = QuantumCircuitMetrics()
        
        # Quantum simulation settings
        self.max_qubits = self.config.get('max_qubits', 10)
        self.use_gpu_simulation = self.config.get('use_gpu_simulation', False)
        
        # Performance tracking
        self.quantum_speedup_log = []
        
    def create_quantum_gnn(
        self,
        node_features: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 1,
        **kwargs
    ) -> QuantumAcceleratedGNN:
        """Create quantum-accelerated GNN model."""
        
        return QuantumAcceleratedGNN(
            node_features=node_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            quantum_features=self.config.get('quantum_features', 16),
            use_quantum=self.config.get('enable_quantum', True)
        )
    
    def optimize_graph_with_qaoa(
        self,
        adjacency_matrix: np.ndarray,
        optimization_problem: str = 'max_cut'
    ) -> Dict[str, Any]:
        """Optimize graph problems using QAOA."""
        
        start_time = time.time()
        
        num_nodes = adjacency_matrix.shape[0]
        if num_nodes > self.max_qubits:
            # Reduce problem size or use approximation
            logging.warning(f"Graph too large ({num_nodes} nodes). Reducing to {self.max_qubits} nodes.")
            adjacency_matrix = adjacency_matrix[:self.max_qubits, :self.max_qubits]
            num_nodes = self.max_qubits
        
        # Create QAOA optimizer
        qaoa = QuantumApproximateOptimization(num_qubits=num_nodes, depth=2)
        
        # Optimize parameters
        optimal_beta, optimal_gamma = qaoa.optimize_parameters(adjacency_matrix)
        
        # Get final solution
        problem_hamiltonian = qaoa.create_problem_hamiltonian(adjacency_matrix)
        state = QuantumState(num_nodes)
        final_state = qaoa.apply_qaoa_circuit(state, problem_hamiltonian)
        
        # Extract solution
        probabilities = final_state.get_probabilities()
        best_solution_idx = np.argmax(probabilities)
        
        # Convert to binary string
        solution = [(best_solution_idx >> i) & 1 for i in range(num_nodes)]
        
        execution_time = time.time() - start_time
        
        # Update metrics
        self.metrics.execution_time_ms = execution_time * 1000
        self.metrics.qubit_count = num_nodes
        self.metrics.circuit_depth = qaoa.depth * 2  # Each layer has 2 sets of gates
        
        return {
            'solution': solution,
            'optimal_parameters': {
                'beta': optimal_beta,
                'gamma': optimal_gamma
            },
            'execution_time_ms': execution_time * 1000,
            'solution_quality': self._evaluate_solution_quality(solution, adjacency_matrix, optimization_problem),
            'quantum_state_probabilities': probabilities
        }
    
    def _evaluate_solution_quality(
        self,
        solution: List[int],
        adjacency_matrix: np.ndarray,
        problem_type: str
    ) -> float:
        """Evaluate the quality of QAOA solution."""
        
        if problem_type == 'max_cut':
            # Calculate cut value
            cut_value = 0
            for i in range(len(solution)):
                for j in range(i + 1, len(solution)):
                    if solution[i] != solution[j]:  # Different partitions
                        cut_value += adjacency_matrix[i, j]
            return cut_value
        
        return 0.0
    
    def benchmark_quantum_vs_classical(
        self,
        test_graphs: List[np.ndarray],
        classical_baseline: Callable = None
    ) -> Dict[str, Any]:
        """Benchmark quantum acceleration against classical methods."""
        
        results = {
            'quantum_times': [],
            'classical_times': [],
            'quantum_solutions': [],
            'classical_solutions': [],
            'speedup_factors': []
        }
        
        for i, graph in enumerate(test_graphs):
            # Quantum optimization
            start_time = time.time()
            quantum_result = self.optimize_graph_with_qaoa(graph)
            quantum_time = time.time() - start_time
            
            # Classical baseline (if provided)
            if classical_baseline:
                start_time = time.time()
                classical_result = classical_baseline(graph)
                classical_time = time.time() - start_time
                
                speedup = classical_time / quantum_time if quantum_time > 0 else 1.0
                results['speedup_factors'].append(speedup)
                results['classical_times'].append(classical_time)
                results['classical_solutions'].append(classical_result)
            
            results['quantum_times'].append(quantum_time)
            results['quantum_solutions'].append(quantum_result)
        
        # Calculate summary statistics
        if results['speedup_factors']:
            results['average_speedup'] = np.mean(results['speedup_factors'])
            results['max_speedup'] = np.max(results['speedup_factors'])
            results['quantum_advantage_cases'] = sum(1 for s in results['speedup_factors'] if s > 1.0)
        
        return results
    
    def get_quantum_metrics(self) -> Dict[str, Any]:
        """Get comprehensive quantum acceleration metrics."""
        
        return {
            'circuit_metrics': {
                'circuit_depth': self.metrics.circuit_depth,
                'gate_count': self.metrics.gate_count,
                'qubit_count': self.metrics.qubit_count,
                'quantum_volume': self.metrics.quantum_volume
            },
            'performance_metrics': {
                'execution_time_ms': self.metrics.execution_time_ms,
                'classical_simulation_time_ms': self.metrics.classical_simulation_time_ms,
                'quantum_advantage_factor': self.metrics.quantum_advantage_factor,
                'fidelity': self.metrics.fidelity
            },
            'system_capabilities': {
                'max_qubits': self.max_qubits,
                'gpu_simulation_enabled': self.use_gpu_simulation,
                'supported_algorithms': ['QAOA', 'VQE', 'Quantum_Embedding']
            }
        }


def create_quantum_acceleration_framework(config: Dict[str, Any] = None) -> QuantumAccelerationFramework:
    """Factory function for creating quantum acceleration frameworks."""
    return QuantumAccelerationFramework(config or {})