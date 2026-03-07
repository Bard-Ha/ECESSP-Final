"""
Enhanced Latent Space Sampler for Generative Battery Discovery

This module provides functionality for:
- Controlled latent space sampling for generative discovery
- Interpolation/extrapolation in latent space conditioned on target properties
- Novelty scoring based on distance from training manifold
- Ranking generated batteries by property alignment + novelty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class LatentSampler:
    """
    Enhanced latent space sampler for generative battery discovery.
    
    Features:
    - Target-conditioned sampling
    - Diversity-controlled generation
    - Novelty scoring
    - Property-guided exploration
    """
    
    def __init__(self, 
                 embedding_dim: int = 128,
                 device: Optional[torch.device] = None,
                 novelty_threshold: float = 0.5):
        """
        Initialize the latent sampler.
        
        Args:
            embedding_dim: Dimension of latent space
            device: Device to use for computations
            novelty_threshold: Threshold for considering a sample novel
        """
        self.embedding_dim = embedding_dim
        self.device = device or torch.device('cpu')
        self.novelty_threshold = novelty_threshold
        
        # Statistics for novelty computation
        self.training_mean = None
        self.training_cov = None
        self.training_std = None
        
        # Property guidance weights
        self.property_weights = {
            'average_voltage': 1.0,
            'capacity_grav': 1.0,
            'capacity_vol': 1.0,
            'energy_grav': 1.0,
            'energy_vol': 1.0,
            'stability_charge': 0.5,
            'stability_discharge': 0.5
        }
        self.property_order = [
            'average_voltage',
            'capacity_grav',
            'capacity_vol',
            'energy_grav',
            'energy_vol',
            'stability_charge',
            'stability_discharge',
        ]
    
    def set_training_statistics(self, 
                               training_embeddings: torch.Tensor,
                               training_properties: Optional[torch.Tensor] = None):
        """
        Set training statistics for novelty computation.
        
        Args:
            training_embeddings: Training embeddings [N, embedding_dim]
            training_properties: Training properties [N, 7] (optional)
        """
        self.training_mean = training_embeddings.mean(dim=0)
        self.training_cov = torch.cov(training_embeddings.T)
        self.training_std = training_embeddings.std(dim=0)
        
        if training_properties is not None:
            # Compute property statistics for guidance
            self.property_mean = training_properties.mean(dim=0)
            self.property_std = training_properties.std(dim=0)
    
    def generate_diverse_samples(self,
                                base_embedding: torch.Tensor,
                                target_objectives: Dict[str, float],
                                num_samples: int = 50,
                                diversity_weight: float = 0.4,
                                extrapolation_strength: float = 0.3) -> torch.Tensor:
        """
        Generate diverse latent samples conditioned on target objectives.
        
        Args:
            base_embedding: Base embedding to start from [embedding_dim]
            target_objectives: Target property objectives
            num_samples: Number of samples to generate
            diversity_weight: Weight for diversity vs objective alignment (0.0-1.0)
            extrapolation_strength: Strength of extrapolation (0.0-1.0)
            
        Returns:
            Generated latent samples [num_samples, embedding_dim]
        """
        if self.training_mean is None:
            raise ValueError("Training statistics not set. Call set_training_statistics first.")
        
        base_embedding = base_embedding.to(self.device)
        
        # Convert target objectives to tensor
        target_tensor = self._objectives_to_tensor(target_objectives).to(self.device)
        
        # Generate base samples around the base embedding
        samples = []
        
        for i in range(num_samples):
            # Sample noise
            noise = torch.randn(self.embedding_dim, device=self.device)
            
            # Compute direction towards target
            target_direction = self._compute_target_direction(target_tensor, base_embedding)
            
            # Combine base direction, noise, and extrapolation
            sample_direction = (
                (1.0 - diversity_weight) * target_direction +
                diversity_weight * noise +
                extrapolation_strength * noise * 0.5
            )
            
            # Normalize and scale
            sample_direction = F.normalize(sample_direction, p=2, dim=0)
            sample_magnitude = torch.norm(base_embedding) * (1.0 + extrapolation_strength * 0.2)
            
            sample = base_embedding + sample_direction * sample_magnitude * 0.1
            samples.append(sample)
        
        return torch.stack(samples)
    
    def _compute_target_direction(self, target_tensor: torch.Tensor, base_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute direction in latent space towards target properties.
        
        Args:
            target_tensor: Target properties [7]
            base_embedding: Base embedding [embedding_dim]
            
        Returns:
            Direction vector [embedding_dim]
        """
        # Simple linear mapping from properties to latent space
        # In practice, this would be learned from training data
        
        # Normalize target properties
        if hasattr(self, 'property_mean') and hasattr(self, 'property_std'):
            target_norm = (target_tensor - self.property_mean.to(self.device)) / self.property_std.to(self.device)
        else:
            target_norm = target_tensor
        
        # Create property-guided direction
        property_direction = torch.zeros(self.embedding_dim, device=self.device)
        
        # Map properties to latent dimensions (simplified mapping)
        for i, (prop_name, weight) in enumerate(self.property_weights.items()):
            if i < len(target_norm):
                # Distribute property influence across latent dimensions
                start_dim = (i * 10) % self.embedding_dim
                end_dim = min(start_dim + 10, self.embedding_dim)
                property_direction[start_dim:end_dim] += target_norm[i] * weight
        
        # Normalize property direction
        if torch.norm(property_direction) > 0:
            property_direction = F.normalize(property_direction, p=2, dim=0)
        
        return property_direction
    
    def _objectives_to_tensor(self, objectives: Dict[str, float]) -> torch.Tensor:
        """Convert objectives dict to tensor."""
        values = []
        for prop in self.property_order:
            values.append(objectives.get(prop, 0.0))
        
        return torch.tensor(values, dtype=torch.float32)

    def _objective_mask_and_weights(
        self,
        *,
        target_objectives: Dict[str, float],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        active = torch.tensor(
            [1.0 if p in target_objectives else 0.0 for p in self.property_order],
            dtype=torch.float32,
            device=device,
        )
        if float(active.sum().item()) <= 0.0:
            active = torch.ones_like(active)

        weights = torch.tensor(
            [self.property_weights.get(p, 1.0) for p in self.property_order],
            dtype=torch.float32,
            device=device,
        )
        weights = weights * active
        weights = weights / weights.sum().clamp(min=1e-6)
        return active, weights

    def _property_scale(self, device: torch.device) -> torch.Tensor:
        if hasattr(self, 'property_std'):
            return self.property_std.to(device).clamp(min=1e-3)
        return torch.tensor(
            [1.0, 120.0, 350.0, 250.0, 900.0, 1.0, 1.0],
            dtype=torch.float32,
            device=device,
        )
    
    def compute_novelty_score(self, latent_samples: torch.Tensor) -> torch.Tensor:
        """
        Compute novelty scores for latent samples.
        
        Args:
            latent_samples: Latent samples [N, embedding_dim]
            
        Returns:
            Novelty scores [N]
        """
        if self.training_mean is None:
            # If no training stats, return uniform novelty
            return torch.ones(latent_samples.size(0), device=latent_samples.device)
        
        # Compute Mahalanobis distance from training distribution
        centered_samples = latent_samples - self.training_mean.to(latent_samples.device)
        
        # Inverse covariance matrix
        try:
            inv_cov = torch.inverse(self.training_cov.to(latent_samples.device))
        except torch.linalg.LinAlgError:
            # Fallback to identity if covariance is singular
            inv_cov = torch.eye(self.embedding_dim, device=latent_samples.device)
        
        # Mahalanobis distance
        mahal_distances = torch.sqrt(torch.sum(centered_samples @ inv_cov * centered_samples, dim=1))
        
        # Convert to novelty score (higher distance = higher novelty)
        # Use tanh to ensure scores are in [0, 1] range
        novelty_scores = (torch.tanh(mahal_distances * 0.5) + 1.0) / 2.0
        
        return novelty_scores
    
    def compute_property_alignment(self, 
                                  latent_samples: torch.Tensor,
                                  target_objectives: Dict[str, float],
                                  decoder: nn.Module) -> torch.Tensor:
        """
        Compute property alignment scores for latent samples.
        
        Args:
            latent_samples: Latent samples [N, embedding_dim]
            target_objectives: Target property objectives
            decoder: Decoder model to predict properties
            
        Returns:
            Alignment scores [N]
        """
        # Predict properties for samples
        with torch.no_grad():
            predicted_properties = decoder(latent_samples)
        
        # Convert target to tensor
        target_tensor = self._objectives_to_tensor(target_objectives).to(latent_samples.device)
        _, weights = self._objective_mask_and_weights(
            target_objectives=target_objectives,
            device=latent_samples.device,
        )
        scale = self._property_scale(latent_samples.device)

        pred_norm = predicted_properties / scale
        target_norm = target_tensor.expand_as(predicted_properties) / scale

        sq_err = (pred_norm - target_norm) ** 2
        alignment = -(sq_err * weights.unsqueeze(0)).sum(dim=1)
        
        return alignment

    def optimize_latent_samples(
        self,
        *,
        latent_samples: torch.Tensor,
        target_objectives: Dict[str, float],
        decoder: nn.Module,
        steps: int = 24,
        lr: float = 0.05,
        anchor_weight: float = 0.08,
        manifold_weight: float = 0.02,
        diversity_preservation: float = 0.06,
    ) -> torch.Tensor:
        """
        Refine latent samples via gradient descent on decoder loss.
        This provides a learned target-to-latent mapping without training a new projector.
        """
        if steps <= 0:
            return latent_samples

        device = latent_samples.device
        target = self._objectives_to_tensor(target_objectives).to(device)
        _, weights = self._objective_mask_and_weights(
            target_objectives=target_objectives,
            device=device,
        )
        scale = self._property_scale(device)

        initial = latent_samples.detach()
        z = initial.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=lr)

        mean_ref = None
        if self.training_mean is not None:
            mean_ref = self.training_mean.to(device)

        decoder.eval()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            pred = decoder(z)
            pred_norm = pred / scale
            target_norm = target.unsqueeze(0).expand_as(pred) / scale
            sq_err = (pred_norm - target_norm) ** 2
            objective_loss = (sq_err * weights.unsqueeze(0)).sum(dim=1).mean()

            anchor_loss = ((z - initial) ** 2).mean()
            if mean_ref is not None:
                manifold_loss = ((z - mean_ref.unsqueeze(0)) ** 2).mean()
            else:
                manifold_loss = torch.tensor(0.0, device=device)

            # Encourage non-collapse: keep variance across candidate latent points.
            diversity_gain = z.var(dim=0, unbiased=False).mean()
            loss = (
                objective_loss
                + anchor_weight * anchor_loss
                + manifold_weight * manifold_loss
                - diversity_preservation * diversity_gain
            )
            loss.backward()
            optimizer.step()

        return z.detach()
    
    def rank_samples(self,
                    latent_samples: torch.Tensor,
                    target_objectives: Dict[str, float],
                    decoder: nn.Module,
                    novelty_weight: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Rank samples by combined novelty and property alignment.
        
        Args:
            latent_samples: Latent samples [N, embedding_dim]
            target_objectives: Target property objectives
            decoder: Decoder model
            novelty_weight: Weight for novelty in ranking (0.0-1.0)
            
        Returns:
            Tuple of (ranked_indices, novelty_scores, alignment_scores)
        """
        # Compute novelty scores
        novelty_scores = self.compute_novelty_score(latent_samples)
        
        # Compute property alignment scores
        alignment_scores = self.compute_property_alignment(latent_samples, target_objectives, decoder)
        
        # Combine scores
        combined_scores = (1.0 - novelty_weight) * alignment_scores + novelty_weight * novelty_scores
        
        # Rank by combined scores
        ranked_indices = torch.argsort(combined_scores, descending=True)
        
        return ranked_indices, novelty_scores, alignment_scores
    
    def sample_from_distribution(self,
                                 mean: torch.Tensor,
                                 std: torch.Tensor,
                                 num_samples: int = 50) -> torch.Tensor:
        """
        Sample from a Gaussian distribution in latent space.
        
        Args:
            mean: Mean of distribution [embedding_dim]
            std: Standard deviation of distribution [embedding_dim]
            num_samples: Number of samples to generate
            
        Returns:
            Samples [num_samples, embedding_dim]
        """
        noise = torch.randn(num_samples, self.embedding_dim, device=self.device)
        samples = mean.unsqueeze(0) + noise * std.unsqueeze(0)
        return samples
    
    def interpolate_between(self,
                           start_embedding: torch.Tensor,
                           end_embedding: torch.Tensor,
                           num_points: int = 10) -> torch.Tensor:
        """
        Interpolate between two embeddings in latent space.
        
        Args:
            start_embedding: Start embedding [embedding_dim]
            end_embedding: End embedding [embedding_dim]
            num_points: Number of interpolation points
            
        Returns:
            Interpolated embeddings [num_points, embedding_dim]
        """
        alphas = torch.linspace(0, 1, num_points, device=self.device)
        interpolated = torch.stack([
            start_embedding * (1 - alpha) + end_embedding * alpha
            for alpha in alphas
        ])
        return interpolated
    
    def extrapolate_from(self,
                        base_embedding: torch.Tensor,
                        direction: torch.Tensor,
                        steps: int = 5,
                        step_size: float = 0.1) -> torch.Tensor:
        """
        Extrapolate from a base embedding in a given direction.
        
        Args:
            base_embedding: Base embedding [embedding_dim]
            direction: Direction vector [embedding_dim]
            steps: Number of extrapolation steps
            step_size: Size of each step
            
        Returns:
            Extrapolated embeddings [steps, embedding_dim]
        """
        direction = F.normalize(direction, p=2, dim=0)
        extrapolated = torch.stack([
            base_embedding + direction * step_size * (i + 1)
            for i in range(steps)
        ])
        return extrapolated


class GenerativeDiscoveryEngine:
    """
    High-level interface for generative battery discovery.
    """
    
    def __init__(self, latent_sampler: LatentSampler, decoder: nn.Module):
        self.latent_sampler = latent_sampler
        self.decoder = decoder
    
    def discover_novel_batteries(self,
                                base_system: torch.Tensor,
                                target_objectives: Dict[str, float],
                                num_candidates: int = 100,
                                diversity_weight: float = 0.4,
                                novelty_weight: float = 0.3,
                                extrapolation_strength: float = 0.3,
                                optimize_steps: int = 24,
                                decoder_override: Optional[nn.Module] = None) -> Dict:
        """
        Discover novel battery systems using latent space sampling.
        
        Args:
            base_system: Base system embedding [embedding_dim]
            target_objectives: Target property objectives
            num_candidates: Number of candidate systems to generate
            diversity_weight: Weight for diversity vs objective alignment
            novelty_weight: Weight for novelty in final ranking
            extrapolation_strength: Strength of extrapolation
            
        Returns:
            Dictionary with generated systems and their properties
        """
        decoder = decoder_override if decoder_override is not None else self.decoder
        if decoder is None:
            raise RuntimeError("No decoder available for generative discovery")

        # Generate diverse samples
        latent_samples = self.latent_sampler.generate_diverse_samples(
            base_embedding=base_system,
            target_objectives=target_objectives,
            num_samples=num_candidates,
            diversity_weight=diversity_weight,
            extrapolation_strength=extrapolation_strength
        )

        # Decoder-guided refinement in latent space for stronger target matching.
        latent_samples = self.latent_sampler.optimize_latent_samples(
            latent_samples=latent_samples,
            target_objectives=target_objectives,
            decoder=decoder,
            steps=optimize_steps,
            diversity_preservation=0.04 + 0.10 * float(diversity_weight),
        )
        
        # Rank samples
        ranked_indices, novelty_scores, alignment_scores = self.latent_sampler.rank_samples(
            latent_samples=latent_samples,
            target_objectives=target_objectives,
            decoder=decoder,
            novelty_weight=novelty_weight
        )
        
        # Get top candidates
        top_indices = ranked_indices[:min(50, len(ranked_indices))]
        top_samples = latent_samples[top_indices]
        
        # Decode to properties
        with torch.no_grad():
            predicted_properties = decoder(top_samples)
        
        # Compute novelty statistics
        novelty_mean = novelty_scores[top_indices].mean().item()
        novelty_max = novelty_scores[top_indices].max().item()
        novelty_min = novelty_scores[top_indices].min().item()
        
        return {
            'latent_samples': top_samples,
            'predicted_properties': predicted_properties,
            'novelty_scores': novelty_scores[top_indices],
            'alignment_scores': alignment_scores[top_indices],
            'ranked_indices': top_indices,
            'novelty_statistics': {
                'mean': novelty_mean,
                'max': novelty_max,
                'min': novelty_min,
                'threshold': self.latent_sampler.novelty_threshold
            },
            'generation_params': {
                'num_candidates': num_candidates,
                'diversity_weight': diversity_weight,
                'novelty_weight': novelty_weight,
                'extrapolation_strength': extrapolation_strength,
                'optimize_steps': optimize_steps,
            }
        }
