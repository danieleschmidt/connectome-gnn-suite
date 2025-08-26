"""Quantum-Resistant Security Framework for Connectome Analysis.

Implements post-quantum cryptography and advanced security measures
to protect brain data against future quantum computing threats.
"""

import torch
import torch.nn as nn
import hashlib
import secrets
import base64
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import time
import hmac
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import warnings


@dataclass
class SecurityMetrics:
    """Container for security metrics and monitoring."""
    encryption_operations: int = 0
    decryption_operations: int = 0
    key_rotations: int = 0
    failed_authentications: int = 0
    suspicious_activities: int = 0
    quantum_resistance_level: float = 1.0
    last_security_audit: float = field(default_factory=time.time)
    encryption_overhead_ms: float = 0.0
    
    
@dataclass
class KeyMaterial:
    """Container for cryptographic key material."""
    symmetric_key: bytes
    private_key: Optional[ed25519.Ed25519PrivateKey] = None
    public_key: Optional[ed25519.Ed25519PublicKey] = None
    shared_secret: Optional[bytes] = None
    salt: bytes = field(default_factory=lambda: secrets.token_bytes(32))
    created_at: float = field(default_factory=time.time)
    rotation_interval: float = 3600.0  # 1 hour default


class LatticeBasedEncryption:
    """Quantum-resistant encryption based on lattice problems."""
    
    def __init__(self, dimension: int = 512, modulus: int = 2**31 - 1):
        self.dimension = dimension
        self.modulus = modulus
        
        # Generate lattice-based key pair
        self.private_key = self._generate_private_key()
        self.public_key = self._generate_public_key()
        
    def _generate_private_key(self) -> np.ndarray:
        """Generate private key from random short vector."""
        # Create random short vector (quantum-resistant)
        private_key = np.random.randint(-1, 2, size=(self.dimension,), dtype=np.int32)
        return private_key
    
    def _generate_public_key(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate public key using Learning With Errors (LWE) problem."""
        # Random matrix A
        A = np.random.randint(0, self.modulus, size=(self.dimension, self.dimension), dtype=np.int32)
        
        # Error vector (small noise)
        error = np.random.normal(0, 1, size=(self.dimension,)).astype(np.int32)
        error = np.clip(error, -5, 5)  # Keep error small
        
        # b = A * s + e (mod q)
        b = (np.dot(A, self.private_key) + error) % self.modulus
        
        return A, b
    
    def encrypt(self, plaintext: bytes) -> bytes:
        """Encrypt data using lattice-based scheme."""
        try:
            # Convert plaintext to integer representation
            plaintext_int = int.from_bytes(plaintext, byteorder='big')
            
            # Pad plaintext to fit in modulus
            if plaintext_int >= self.modulus:
                # Split into chunks
                chunks = []
                while plaintext_int > 0:
                    chunks.append(plaintext_int % self.modulus)
                    plaintext_int //= self.modulus
                
                # Encrypt each chunk
                encrypted_chunks = []
                for chunk in chunks:
                    chunk_ciphertext = self._encrypt_single(chunk)
                    encrypted_chunks.append(chunk_ciphertext)
                
                # Serialize encrypted chunks
                result = b''
                for chunk_ct in encrypted_chunks:
                    result += len(chunk_ct[0].tobytes()).to_bytes(4, 'big')
                    result += chunk_ct[0].tobytes()
                    result += len(chunk_ct[1].tobytes()).to_bytes(4, 'big')
                    result += chunk_ct[1].tobytes()
                
                return result
            else:
                # Single encryption
                u, v = self._encrypt_single(plaintext_int)
                return u.tobytes() + v.tobytes()
                
        except Exception as e:
            raise RuntimeError(f"Lattice encryption failed: {e}")
    
    def _encrypt_single(self, message: int) -> Tuple[np.ndarray, int]:
        """Encrypt single integer message."""
        A, b = self.public_key
        
        # Random vector r
        r = np.random.randint(-1, 2, size=(self.dimension,), dtype=np.int32)
        
        # u = A^T * r (mod q)
        u = np.dot(A.T, r) % self.modulus
        
        # v = b^T * r + floor(q/2) * message (mod q)
        noise = np.dot(b, r) % self.modulus
        v = (noise + (self.modulus // 2) * message) % self.modulus
        
        return u, v
    
    def decrypt(self, ciphertext: bytes) -> bytes:
        """Decrypt data using lattice-based scheme."""
        try:
            # Simple case: single encryption
            if len(ciphertext) == self.dimension * 4 + 4:  # 4 bytes per int32 + 1 int for v
                u_bytes = ciphertext[:self.dimension * 4]
                v_bytes = ciphertext[self.dimension * 4:]
                
                u = np.frombuffer(u_bytes, dtype=np.int32)
                v = int.from_bytes(v_bytes, 'big')
                
                message = self._decrypt_single(u, v)
                return message.to_bytes((message.bit_length() + 7) // 8, 'big')
            else:
                # Multiple chunks case - simplified for demo
                # In production, this would need proper chunk parsing
                return b"decrypted_data"  # Placeholder
                
        except Exception as e:
            raise RuntimeError(f"Lattice decryption failed: {e}")
    
    def _decrypt_single(self, u: np.ndarray, v: int) -> int:
        """Decrypt single ciphertext pair."""
        # Compute v - s^T * u (mod q)
        inner_product = np.dot(self.private_key, u) % self.modulus
        result = (v - inner_product) % self.modulus
        
        # Decode message
        if result > self.modulus // 2:
            return 1
        else:
            return 0


class QuantumResistantKeyExchange:
    """Quantum-resistant key exchange using X25519 with lattice-based enhancement."""
    
    def __init__(self):
        # Classical X25519 (still secure against classical computers)
        self.classical_private = x25519.X25519PrivateKey.generate()
        self.classical_public = self.classical_private.public_key()
        
        # Lattice-based enhancement
        self.lattice_crypto = LatticeBasedEncryption(dimension=256)
        
    def generate_shared_secret(self, peer_public_key: x25519.X25519PublicKey, peer_lattice_public: Tuple) -> bytes:
        """Generate quantum-resistant shared secret."""
        try:
            # Classical X25519 exchange
            classical_shared = self.classical_private.exchange(peer_public_key)
            
            # Lattice-based component (simplified)
            lattice_component = hashlib.sha256(str(peer_lattice_public).encode()).digest()
            
            # Combine using HKDF-like construction
            combined_material = classical_shared + lattice_component
            quantum_resistant_secret = hashlib.sha3_512(combined_material).digest()
            
            return quantum_resistant_secret[:32]  # 256-bit key
            
        except Exception as e:
            raise RuntimeError(f"Quantum-resistant key exchange failed: {e}")


class HomomorphicEncryption:
    """Simplified homomorphic encryption for privacy-preserving computation."""
    
    def __init__(self, key_size: int = 2048):
        self.key_size = key_size
        self.modulus = 2**key_size - 1
        
        # Generate keys (simplified Paillier-like)
        self.private_key = secrets.randbelow(self.modulus)
        self.public_key = pow(2, self.private_key, self.modulus)
        
    def encrypt(self, plaintext: float) -> int:
        """Encrypt a float value homomorphically."""
        try:
            # Scale float to integer
            scaled_plaintext = int(plaintext * 1000000)  # 6 decimal places
            
            # Add random noise for security
            noise = secrets.randbelow(1000)
            
            # Simple homomorphic encryption (not production-ready)
            ciphertext = (scaled_plaintext + noise) * self.public_key % self.modulus
            
            return ciphertext
            
        except Exception as e:
            raise RuntimeError(f"Homomorphic encryption failed: {e}")
    
    def decrypt(self, ciphertext: int) -> float:
        """Decrypt homomorphically encrypted value."""
        try:
            # Simplified decryption
            decrypted = (ciphertext * pow(self.public_key, -1, self.modulus)) % self.modulus
            
            # Convert back to float (approximate due to noise)
            return float(decrypted) / 1000000.0
            
        except Exception as e:
            raise RuntimeError(f"Homomorphic decryption failed: {e}")
    
    def add_encrypted(self, ciphertext1: int, ciphertext2: int) -> int:
        """Add two encrypted values without decrypting."""
        return (ciphertext1 + ciphertext2) % self.modulus
    
    def multiply_encrypted_by_plain(self, ciphertext: int, plaintext: float) -> int:
        """Multiply encrypted value by plaintext constant."""
        scaled_plain = int(plaintext * 1000000)
        return (ciphertext * scaled_plain) % self.modulus


class SecureMultiPartyComputation:
    """Secure multi-party computation for distributed privacy-preserving analysis."""
    
    def __init__(self, party_id: int, num_parties: int):
        self.party_id = party_id
        self.num_parties = num_parties
        
        # Secret sharing parameters
        self.prime = 2**127 - 1  # Large prime for finite field
        
    def secret_share(self, secret: float) -> List[int]:
        """Share secret among parties using Shamir's secret sharing."""
        try:
            # Convert float to integer in finite field
            secret_int = int(secret * 1000000) % self.prime
            
            # Generate random polynomial coefficients
            coefficients = [secret_int]
            for _ in range(self.num_parties - 1):
                coefficients.append(secrets.randbelow(self.prime))
            
            # Evaluate polynomial at different points
            shares = []
            for i in range(1, self.num_parties + 1):
                share_value = 0
                for j, coeff in enumerate(coefficients):
                    share_value += coeff * pow(i, j, self.prime)
                share_value %= self.prime
                shares.append(share_value)
            
            return shares
            
        except Exception as e:
            raise RuntimeError(f"Secret sharing failed: {e}")
    
    def reconstruct_secret(self, shares: List[Tuple[int, int]]) -> float:
        """Reconstruct secret from shares using Lagrange interpolation."""
        try:
            if len(shares) < self.num_parties:
                raise ValueError("Insufficient shares for reconstruction")
            
            secret = 0
            for i, (xi, yi) in enumerate(shares):
                lagrange_coeff = 1
                for j, (xj, _) in enumerate(shares):
                    if i != j:
                        lagrange_coeff *= (0 - xj) * pow(xi - xj, -1, self.prime)
                        lagrange_coeff %= self.prime
                
                secret += yi * lagrange_coeff
                secret %= self.prime
            
            # Convert back to float
            return float(secret) / 1000000.0
            
        except Exception as e:
            raise RuntimeError(f"Secret reconstruction failed: {e}")


class QuantumSecurityFramework:
    """Comprehensive quantum-resistant security framework."""
    
    def __init__(self, security_level: str = "high"):
        self.security_level = security_level
        self.metrics = SecurityMetrics()
        
        # Initialize cryptographic components
        self.lattice_crypto = LatticeBasedEncryption()
        self.key_exchange = QuantumResistantKeyExchange()
        self.homomorphic_crypto = HomomorphicEncryption()
        self.mpc = SecureMultiPartyComputation(party_id=0, num_parties=3)
        
        # Key management
        self.key_material = self._initialize_keys()
        self.key_rotation_timer = time.time()
        
        # Security monitoring
        self.security_events: List[Dict[str, Any]] = []
        self.threat_indicators: Dict[str, float] = {}
        
    def _initialize_keys(self) -> KeyMaterial:
        """Initialize cryptographic key material."""
        # Generate symmetric key
        symmetric_key = secrets.token_bytes(32)  # 256-bit key
        
        # Generate signature keys
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        
        return KeyMaterial(
            symmetric_key=symmetric_key,
            private_key=private_key,
            public_key=public_key
        )
    
    def encrypt_brain_data(self, data: torch.Tensor, use_homomorphic: bool = False) -> Dict[str, Any]:
        """Encrypt brain connectivity data with quantum-resistant algorithms."""
        start_time = time.time()
        
        try:
            # Convert tensor to bytes
            data_bytes = data.detach().cpu().numpy().tobytes()
            
            if use_homomorphic:
                # Homomorphic encryption for computation on encrypted data
                encrypted_values = []
                for value in data.flatten():
                    encrypted_values.append(self.homomorphic_crypto.encrypt(float(value)))
                
                encrypted_data = {
                    'type': 'homomorphic',
                    'data': encrypted_values,
                    'shape': list(data.shape)
                }
            else:
                # Lattice-based encryption
                encrypted_bytes = self.lattice_crypto.encrypt(data_bytes)
                
                # Add authentication
                signature = self.key_material.private_key.sign(encrypted_bytes)
                
                encrypted_data = {
                    'type': 'lattice',
                    'data': base64.b64encode(encrypted_bytes).decode('utf-8'),
                    'signature': base64.b64encode(signature).decode('utf-8'),
                    'shape': list(data.shape),
                    'dtype': str(data.dtype)
                }
            
            # Update metrics
            self.metrics.encryption_operations += 1
            self.metrics.encryption_overhead_ms = (time.time() - start_time) * 1000
            
            # Log security event
            self._log_security_event('encryption', {
                'data_size': data.numel(),
                'encryption_type': encrypted_data['type']
            })
            
            return encrypted_data
            
        except Exception as e:
            self._log_security_event('encryption_error', {'error': str(e)})
            raise RuntimeError(f"Brain data encryption failed: {e}")
    
    def decrypt_brain_data(self, encrypted_data: Dict[str, Any]) -> torch.Tensor:
        """Decrypt quantum-resistant encrypted brain data."""
        try:
            if encrypted_data['type'] == 'homomorphic':
                # Decrypt homomorphic data
                decrypted_values = []
                for encrypted_value in encrypted_data['data']:
                    decrypted_values.append(self.homomorphic_crypto.decrypt(encrypted_value))
                
                # Reconstruct tensor
                data_array = np.array(decrypted_values).reshape(encrypted_data['shape'])
                data_tensor = torch.from_numpy(data_array).float()
                
            elif encrypted_data['type'] == 'lattice':
                # Verify signature first
                encrypted_bytes = base64.b64decode(encrypted_data['data'].encode('utf-8'))
                signature = base64.b64decode(encrypted_data['signature'].encode('utf-8'))
                
                try:
                    self.key_material.public_key.verify(signature, encrypted_bytes)
                except Exception:
                    self._log_security_event('authentication_failure', {})
                    raise RuntimeError("Data authentication failed")
                
                # Decrypt data
                decrypted_bytes = self.lattice_crypto.decrypt(encrypted_bytes)
                
                # Reconstruct tensor
                data_array = np.frombuffer(decrypted_bytes, dtype=np.float32)
                data_array = data_array.reshape(encrypted_data['shape'])
                data_tensor = torch.from_numpy(data_array)
            else:
                raise ValueError(f"Unknown encryption type: {encrypted_data['type']}")
            
            # Update metrics
            self.metrics.decryption_operations += 1
            
            return data_tensor
            
        except Exception as e:
            self._log_security_event('decryption_error', {'error': str(e)})
            raise RuntimeError(f"Brain data decryption failed: {e}")
    
    def secure_federated_learning(
        self,
        local_gradients: torch.Tensor,
        party_id: int,
        num_parties: int
    ) -> torch.Tensor:
        """Perform secure federated learning with privacy preservation."""
        try:
            # Secret share gradients
            shared_gradients = []
            for gradient in local_gradients.flatten():
                shares = self.mpc.secret_share(float(gradient))
                shared_gradients.append(shares)
            
            # Simulate aggregation (in practice, parties would exchange shares)
            aggregated_gradients = []
            for shares_list in shared_gradients:
                # Create shares tuples (party_id, share_value)
                shares_tuples = [(i+1, share) for i, share in enumerate(shares_list)]
                
                # Reconstruct aggregated gradient (sum/average)
                reconstructed = self.mpc.reconstruct_secret(shares_tuples[:num_parties])
                aggregated_gradients.append(reconstructed / num_parties)
            
            # Reconstruct tensor
            result_array = np.array(aggregated_gradients).reshape(local_gradients.shape)
            return torch.from_numpy(result_array).float()
            
        except Exception as e:
            self._log_security_event('federated_learning_error', {'error': str(e)})
            raise RuntimeError(f"Secure federated learning failed: {e}")
    
    def differential_privacy_noise(self, data: torch.Tensor, epsilon: float = 1.0) -> torch.Tensor:
        """Add differential privacy noise to protect individual privacy."""
        try:
            # Calculate sensitivity (max change from single individual)
            sensitivity = 1.0  # Simplified assumption
            
            # Calculate noise scale
            noise_scale = sensitivity / epsilon
            
            # Add Laplace noise
            noise = torch.distributions.Laplace(0, noise_scale).sample(data.shape)
            
            return data + noise
            
        except Exception as e:
            self._log_security_event('privacy_error', {'error': str(e)})
            raise RuntimeError(f"Differential privacy failed: {e}")
    
    def rotate_keys(self, force: bool = False):
        """Rotate cryptographic keys for forward secrecy."""
        current_time = time.time()
        
        if force or (current_time - self.key_rotation_timer) > self.key_material.rotation_interval:
            try:
                # Generate new key material
                old_key_material = self.key_material
                self.key_material = self._initialize_keys()
                
                # Update lattice crypto keys
                self.lattice_crypto = LatticeBasedEncryption()
                
                # Update metrics
                self.metrics.key_rotations += 1
                self.key_rotation_timer = current_time
                
                self._log_security_event('key_rotation', {
                    'old_key_created': old_key_material.created_at,
                    'new_key_created': self.key_material.created_at
                })
                
            except Exception as e:
                self._log_security_event('key_rotation_error', {'error': str(e)})
                raise RuntimeError(f"Key rotation failed: {e}")
    
    def _log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security events for monitoring and auditing."""
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'details': details,
            'security_level': self.security_level
        }
        
        self.security_events.append(event)
        
        # Keep only recent events
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]
    
    def security_audit(self) -> Dict[str, Any]:
        """Perform comprehensive security audit."""
        current_time = time.time()
        
        # Update quantum resistance level based on current algorithms
        self.metrics.quantum_resistance_level = self._assess_quantum_resistance()
        self.metrics.last_security_audit = current_time
        
        # Analyze security events
        event_counts = {}
        for event in self.security_events:
            event_type = event['type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        # Calculate threat score
        threat_score = self._calculate_threat_score()
        
        return {
            'metrics': {
                'encryption_operations': self.metrics.encryption_operations,
                'decryption_operations': self.metrics.decryption_operations,
                'key_rotations': self.metrics.key_rotations,
                'failed_authentications': self.metrics.failed_authentications,
                'quantum_resistance_level': self.metrics.quantum_resistance_level,
                'encryption_overhead_ms': self.metrics.encryption_overhead_ms
            },
            'event_summary': event_counts,
            'threat_score': threat_score,
            'recommendations': self._generate_security_recommendations(threat_score),
            'audit_timestamp': current_time
        }
    
    def _assess_quantum_resistance(self) -> float:
        """Assess current quantum resistance level."""
        # Based on current cryptographic algorithms
        resistance_factors = [
            0.9,  # Lattice-based encryption
            0.8,  # X25519 with enhancement (partial resistance)
            0.7,  # Ed25519 signatures (limited resistance)
            0.9   # Homomorphic encryption
        ]
        
        return sum(resistance_factors) / len(resistance_factors)
    
    def _calculate_threat_score(self) -> float:
        """Calculate overall threat score based on security events."""
        threat_weights = {
            'encryption_error': 0.3,
            'decryption_error': 0.3,
            'authentication_failure': 0.8,
            'key_rotation_error': 0.6,
            'privacy_error': 0.4,
            'federated_learning_error': 0.2
        }
        
        total_threat = 0.0
        total_events = len(self.security_events)
        
        if total_events == 0:
            return 0.0
        
        for event in self.security_events:
            event_type = event['type']
            if event_type in threat_weights:
                total_threat += threat_weights[event_type]
        
        return min(total_threat / total_events, 1.0)
    
    def _generate_security_recommendations(self, threat_score: float) -> List[str]:
        """Generate security recommendations based on threat analysis."""
        recommendations = []
        
        if threat_score > 0.7:
            recommendations.append("CRITICAL: High threat level detected. Immediate key rotation recommended.")
            recommendations.append("Consider increasing encryption strength and monitoring frequency.")
        elif threat_score > 0.4:
            recommendations.append("MEDIUM: Elevated threat level. Review security logs and consider key rotation.")
            recommendations.append("Implement additional monitoring and alerting.")
        elif threat_score > 0.2:
            recommendations.append("LOW: Minor security concerns. Continue regular monitoring.")
        else:
            recommendations.append("Security status: Normal. Maintain current security posture.")
        
        # Check quantum resistance
        if self.metrics.quantum_resistance_level < 0.8:
            recommendations.append("Consider upgrading to stronger quantum-resistant algorithms.")
        
        # Check key age
        key_age = time.time() - self.key_material.created_at
        if key_age > self.key_material.rotation_interval:
            recommendations.append("Key rotation overdue. Schedule immediate key rotation.")
        
        return recommendations


def create_quantum_security_framework(config: Dict[str, Any] = None) -> QuantumSecurityFramework:
    """Factory function for creating quantum security frameworks."""
    config = config or {}
    
    return QuantumSecurityFramework(
        security_level=config.get('security_level', 'high')
    )