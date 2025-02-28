
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, List
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# Install required packages if needed
try:
    import sklearn.metrics
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "scikit-learn", "transformers", "datasets"])
    import sklearn.metrics


class NaturalSparseAttention(nn.Module):
    """
    Implementation of Natural Sparse Attention as described in the paper.
    """
    def __init__(
        self,
        hidden_size: int, 
        num_heads: int,
        block_size: int = 16,
        stride: int = 8,
        window_size: int = 128,
        selection_size: int = 128,
        dropout: float = 0.1,
        is_decoder: bool = True,
        is_gqa: bool = True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.block_size = block_size
        self.stride = stride
        self.window_size = window_size
        self.selection_size = selection_size
        self.is_decoder = is_decoder
        self.is_gqa = is_gqa
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        
        # Separate projections for each branch
        self.k_cmp_proj = nn.Linear(hidden_size, hidden_size)
        self.v_cmp_proj = nn.Linear(hidden_size, hidden_size)
        
        self.k_slc_proj = nn.Linear(hidden_size, hidden_size)
        self.v_slc_proj = nn.Linear(hidden_size, hidden_size)
        
        self.k_win_proj = nn.Linear(hidden_size, hidden_size)
        self.v_win_proj = nn.Linear(hidden_size, hidden_size)
        
        # Block compressor MLP as per paper section 3.3.1
        # Using a more complex MLP with position embeddings
        self.block_compressor = nn.Sequential(
            nn.Linear(block_size * self.head_dim, 4 * self.head_dim),
            nn.GELU(),
            nn.Linear(4 * self.head_dim, self.head_dim)
        )
        
        # Position encodings for compression blocks
        self.pos_encodings = nn.Parameter(torch.randn(1, block_size, self.head_dim))
        
        # Gating mechanism
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_size, 3),
            nn.Sigmoid()
        )
        
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def _compress_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compress token blocks into block-level representations as per section 3.3.1
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Compressed tensor of shape (batch_size, num_blocks, hidden_size)
        """
        batch_size, seq_len, hidden_size = x.shape
        device = x.device
        
        # Calculate number of blocks
        num_blocks = max(1, (seq_len - self.block_size) // self.stride + 1)
        
        # Initialize compressed tokens tensor
        compressed = torch.zeros(batch_size, num_blocks, hidden_size, device=device)
        
        # Reshape x for multi-head processing
        x_reshaped = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        for i in range(num_blocks):
            start_idx = i * self.stride
            end_idx = min(start_idx + self.block_size, seq_len)
            if end_idx <= start_idx:
                continue
                
            # Extract block
            block = x_reshaped[:, start_idx:end_idx, :, :]  # (batch_size, block_size*, num_heads, head_dim)
            
            # Apply position encodings to each head dimension
            if end_idx - start_idx < self.block_size:
                # Handle partial blocks
                pos_enc = self.pos_encodings[:, :end_idx-start_idx, :].expand(batch_size, -1, -1)
                block = block + pos_enc.unsqueeze(2)  # Add position encoding
                
                # Pad to full block size
                padding = torch.zeros(
                    batch_size, 
                    self.block_size - (end_idx - start_idx), 
                    self.num_heads,
                    self.head_dim, 
                    device=device
                )
                block = torch.cat([block, padding], dim=1)
            else:
                # Full block with position encodings
                pos_enc = self.pos_encodings.expand(batch_size, -1, -1)
                block = block + pos_enc.unsqueeze(2)  # Add position encoding
            
            # Process each head separately
            for h in range(self.num_heads):
                head_block = block[:, :, h, :]  # (batch_size, block_size, head_dim)
                
                # Flatten the block
                flat_block = head_block.reshape(batch_size, -1)  # (batch_size, block_size*head_dim)
                
                # Apply MLP compression
                compressed_head = self.block_compressor(flat_block)  # (batch_size, head_dim)
                
                # Store in the right position
                compressed[:, i, h*self.head_dim:(h+1)*self.head_dim] = compressed_head
            
        return compressed
        
    def _select_tokens(self, 
                      q: torch.Tensor, 
                      k: torch.Tensor, 
                      v: torch.Tensor, 
                      cmp_k: torch.Tensor, 
                      cmp_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select the most important tokens based on compressed attention scores
        as per section 3.3.2 of the paper
        
        Args:
            q: Query tensor (batch_size, num_heads, 1, head_dim)
            k: Keys tensor (batch_size, num_heads, seq_len, head_dim)
            v: Values tensor (batch_size, num_heads, seq_len, head_dim)
            cmp_k: Compressed keys (batch_size, num_heads, num_blocks, head_dim)
            cmp_scores: Compressed attention scores (batch_size, num_heads, num_blocks)
            
        Returns:
            Tuple of selected keys and values
        """
        batch_size, num_heads, seq_len, head_dim = k.shape
        device = k.device
        
        # Determine selection block size (l' in the paper)
        selection_block_size = self.block_size
        
        # Initialize block scores
        num_selection_blocks = max(1, seq_len // selection_block_size)
        block_scores = torch.zeros(batch_size, num_heads, num_selection_blocks, device=device)
        
        # Compute block importance scores as per equation (9) in the paper
        # Here we're assuming the compression block size equals selection block size for simplicity
        for i in range(min(num_selection_blocks, cmp_scores.size(2))):
            block_scores[:, :, i] = cmp_scores[:, :, i]
        
        # For GQA, ensure consistent block selection across heads as per equation (10)
        if self.is_gqa:
            # Sum scores across heads
            group_block_scores = block_scores.sum(dim=1, keepdim=True)
            # Get top blocks indices
            _, top_indices = torch.topk(
                group_block_scores.squeeze(1), 
                min(self.selection_size // selection_block_size, num_selection_blocks), 
                dim=-1
            )
        else:
            # Per-head selection
            _, top_indices = torch.topk(
                block_scores, 
                min(self.selection_size // selection_block_size, num_selection_blocks), 
                dim=-1
            )
        
        # Initialize selected key-value tensors
        max_tokens = min(self.selection_size, seq_len)
        selected_k = torch.zeros(batch_size, num_heads, max_tokens, head_dim, device=device)
        selected_v = torch.zeros(batch_size, num_heads, max_tokens, head_dim, device=device)
        
        # Extract selected blocks as per equation (12)
        for batch_idx in range(batch_size):
            for head_idx in range(num_heads):
                # Get indices for this head (or group in GQA)
                head_indices = top_indices[batch_idx, 0 if self.is_gqa else head_idx]
                offset = 0
                
                for block_idx in head_indices:
                    start_pos = block_idx * selection_block_size
                    end_pos = min(start_pos + selection_block_size, seq_len)
                    block_len = end_pos - start_pos
                    
                    if offset + block_len <= max_tokens:
                        selected_k[batch_idx, head_idx, offset:offset+block_len] = k[batch_idx, head_idx, start_pos:end_pos]
                        selected_v[batch_idx, head_idx, offset:offset+block_len] = v[batch_idx, head_idx, start_pos:end_pos]
                        offset += block_len
        
        return selected_k, selected_v
    
    def _get_sliding_window(self, k: torch.Tensor, v: torch.Tensor, current_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract sliding window of recent tokens as per section 3.3.3
        
        Args:
            k: Keys tensor
            v: Values tensor
            current_len: Current sequence length
            
        Returns:
            Window keys and values
        """
        start_pos = max(0, current_len - self.window_size)
        window_k = k[:, :, start_pos:current_len, :]
        window_v = v[:, :, start_pos:current_len, :]
        return window_k, window_v
        
    def forward(self, 
              hidden_states: torch.Tensor, 
              attention_mask: Optional[torch.Tensor] = None,
              past_key_values: Optional[Tuple[torch.Tensor]] = None) -> torch.Tensor:
        """
        Forward pass for Natural Sparse Attention
        
        Args:
            hidden_states: Input tensor (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask (batch_size, seq_len)
            past_key_values: Past key values for incremental decoding
            
        Returns:
            Output tensor (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Project queries, keys, and values
        q = self.q_proj(hidden_states)
        
        # Keys and values for different branches
        k_cmp = self.k_cmp_proj(hidden_states)
        v_cmp = self.v_cmp_proj(hidden_states)
        
        k_slc = self.k_slc_proj(hidden_states)
        v_slc = self.v_slc_proj(hidden_states)
        
        k_win = self.k_win_proj(hidden_states)
        v_win = self.v_win_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        k_cmp = k_cmp.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_cmp = v_cmp.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        k_slc = k_slc.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_slc = v_slc.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        k_win = k_win.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_win = v_win.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute branch gating weights
        gates = self.gate_proj(hidden_states)  # (batch_size, seq_len, 3)
        
        # Normalize gates across branches
        gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Split gates for each branch
        g_cmp = gates[:, :, 0].unsqueeze(1).unsqueeze(1)  # For broadcasting
        g_slc = gates[:, :, 1].unsqueeze(1).unsqueeze(1)
        g_win = gates[:, :, 2].unsqueeze(1).unsqueeze(1)
        
        # Initialize output
        output = torch.zeros_like(hidden_states)
        
        # For causal attention implementation - process each position
        for pos in range(seq_len):
            curr_q = q[:, :, pos:pos+1, :]  # Current query
            
            # Skip compression & selection for first position (no context)
            if pos > 0:
                # 1. Compressed attention
                past_hidden = hidden_states[:, :pos, :]
                
                # Compress past tokens as per section 3.3.1
                compressed = self._compress_tokens(past_hidden)
                
                # Project compressed tokens
                cmp_k_shape = compressed.shape
                cmp_k = compressed.view(
                    cmp_k_shape[0], cmp_k_shape[1], self.num_heads, self.head_dim
                ).transpose(1, 2)
                
                cmp_v = compressed.view(
                    cmp_k_shape[0], cmp_k_shape[1], self.num_heads, self.head_dim
                ).transpose(1, 2)
                
                # Compute attention scores
                cmp_scores = torch.matmul(curr_q, cmp_k.transpose(-1, -2)) / math.sqrt(self.head_dim)
                cmp_attn_weights = F.softmax(cmp_scores, dim=-1)
                cmp_context = torch.matmul(cmp_attn_weights, cmp_v)
                
                # 2. Selection attention as per section 3.3.2
                slc_k, slc_v = self._select_tokens(
                    curr_q, k_slc[:, :, :pos, :], v_slc[:, :, :pos, :], cmp_k, cmp_scores
                )
                
                slc_scores = torch.matmul(curr_q, slc_k.transpose(-1, -2)) / math.sqrt(self.head_dim)
                slc_attn_weights = F.softmax(slc_scores, dim=-1)
                slc_context = torch.matmul(slc_attn_weights, slc_v)
                
                # 3. Window attention as per section 3.3.3
                win_k, win_v = self._get_sliding_window(k_win, v_win, pos)
                
                win_scores = torch.matmul(curr_q, win_k.transpose(-1, -2)) / math.sqrt(self.head_dim)
                win_attn_weights = F.softmax(win_scores, dim=-1)
                win_context = torch.matmul(win_attn_weights, win_v)
                
                # Combine with gating
                combined_context = (
                    g_cmp[:, :, pos:pos+1, :] * cmp_context +
                    g_slc[:, :, pos:pos+1, :] * slc_context +
                    g_win[:, :, pos:pos+1, :] * win_context
                )
            else:
                # For first token, just use self-attention
                curr_k = k_win[:, :, 0:1, :]
                curr_v = v_win[:, :, 0:1, :]
                
                # Simplified attention for first token
                attn_scores = torch.matmul(curr_q, curr_k.transpose(-1, -2)) / math.sqrt(self.head_dim)
                attn_weights = F.softmax(attn_scores, dim=-1)
                combined_context = torch.matmul(attn_weights, curr_v)
            
            # Reshape and place in output
            combined_context = combined_context.transpose(1, 2).reshape(batch_size, 1, self.hidden_size)
            output[:, pos:pos+1, :] = combined_context
        
        # Final projection
        output = self.out_proj(output)
        output = self.dropout(output)
        
        return output


class MoELayer(nn.Module):
    """
    Mixture of Experts layer with routing mechanism
    """
    def __init__(
        self,
        hidden_size: int,
        ffn_dim: int,
        num_experts: int = 8,
        expert_capacity: float = 0.25,
        dropout: float = 0.1,
        gate_k: int = 2,
        router_z_loss_coef: float = 0.001,
        load_balancing_loss_coef: float = 0.01
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.gate_k = gate_k
        self.router_z_loss_coef = router_z_loss_coef
        self.load_balancing_loss_coef = load_balancing_loss_coef
        
        # Create experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, hidden_size),
                nn.Dropout(dropout)
            )
            for _ in range(num_experts)
        ])
        
        # Router
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Initialize router weights
        self.router.weight.data.normal_(mean=0.0, std=0.01)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for MoE layer"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        
        # Reshape for routing
        reshaped_inputs = hidden_states.view(-1, hidden_size)
        total_tokens = batch_size * seq_len
        
        # Calculate routing probabilities
        router_logits = self.router(reshaped_inputs)  # (batch_size * seq_len, num_experts)
        
        # Calculate auxiliary losses
        router_z_loss = self.router_z_loss_coef * (router_logits ** 2).mean()
        
        # Compute dispatch weights
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Calculate load balancing loss
        expert_probs = routing_weights.mean(dim=0)
        target_probs = torch.ones_like(expert_probs) / self.num_experts
        load_balancing_loss = self.load_balancing_loss_coef * \
                               F.mse_loss(expert_probs, target_probs)
        
        # Combined auxiliary loss
        aux_loss = router_z_loss + load_balancing_loss
        
        # Get top-k experts per token
        routing_weights_k, indices_k = torch.topk(routing_weights, k=self.gate_k, dim=-1)
        
        # Normalize top-k weights
        routing_weights_k = routing_weights_k / (routing_weights_k.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Set up expert capacity
        expert_capacity = int(self.expert_capacity * total_tokens)
        
        # Initialize output tensor
        final_output = torch.zeros_like(reshaped_inputs)
        
        # Track tokens per expert
        expert_counts = torch.zeros(self.num_experts, device=device)
        
        # Process tokens with experts
        for token_idx in range(total_tokens):
            for k_idx in range(self.gate_k):
                expert_idx = indices_k[token_idx, k_idx].item()
                weight = routing_weights_k[token_idx, k_idx].item()
                
                # Skip if weight is too small
                if weight < 1e-3:
                    continue
                
                # Process token if expert has capacity
                if expert_counts[expert_idx] < expert_capacity:
                    token_feature = reshaped_inputs[token_idx:token_idx+1]
                    expert_output = self.experts[expert_idx](token_feature)
                    final_output[token_idx] += weight * expert_output.squeeze(0)
                    expert_counts[expert_idx] += 1
        
        # Reshape back to original dimensions
        output = final_output.view(batch_size, seq_len, hidden_size)
        
        return output, aux_loss

class FlashAttention(nn.Module):
    """
    Flash Attention implementation that's compatible with both CUDA and MPS devices.
    Uses tiling strategy to compute attention in a memory-efficient way.
    """
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        
    def forward(self, q, k, v, mask=None):
        """
        Args:
            q, k, v: Query, Key, Value tensors (batch_size, num_heads, seq_len, head_dim)
            mask: Optional attention mask
        
        Returns:
            Output tensor after attention (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, num_heads, seq_len_q, head_dim = q.shape
        _, _, seq_len_k, _ = k.shape
        device = q.device
        
        # Scale factor for attention scores
        scale = 1.0 / math.sqrt(head_dim)
        
        # Check if we can use optimized implementations
        if device.type == 'cuda' and hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # Use PyTorch's native Flash Attention on CUDA
            if mask is not None:
                # Adjust mask dimensions for compatibility
                if mask.dim() == 2:
                    # Convert 2D mask to 4D for multi-head attention
                    mask = mask.unsqueeze(1).unsqueeze(2)
                
                # Convert mask to proper format (True for values to mask)
                mask = mask == 0
                
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=mask is None  # Assume causal if no mask provided
            )
            return output
        
        # Custom tiled implementation for MPS and CPU
        # This avoids materializing the full attention matrix
        
        # Initialize output tensor
        output = torch.zeros_like(q)
        
        # Determine tile size based on memory constraints
        # Smaller tiles for larger head dimensions
        tile_size = min(128, seq_len_k)
        
        # Process queries in sequence_length chunks
        for i in range(0, seq_len_q, tile_size):
            # Get current query block
            q_block = q[:, :, i:min(i+tile_size, seq_len_q), :]
            curr_tile_size = q_block.size(2)
            
            # Process keys and values in sequence_length chunks
            acc_att = torch.zeros((batch_size, num_heads, curr_tile_size, head_dim), device=device)
            acc_normalizer = torch.zeros((batch_size, num_heads, curr_tile_size, 1), device=device)
            
            for j in range(0, seq_len_k, tile_size):
                # Get current key/value blocks
                k_block = k[:, :, j:min(j+tile_size, seq_len_k), :]
                v_block = v[:, :, j:min(j+tile_size, seq_len_k), :]
                curr_k_tile_size = k_block.size(2)
                
                # Calculate attention scores for current blocks
                scores = torch.matmul(q_block, k_block.transpose(-1, -2)) * scale
                
                # Apply mask if provided
                if mask is not None:
                    if mask.dim() == 2:
                        # Expand 2D mask for current blocks
                        mask_block = mask[:, j:min(j+tile_size, seq_len_k)]
                        mask_block = mask_block.unsqueeze(1).unsqueeze(2).expand(
                            -1, num_heads, curr_tile_size, -1
                        )
                    else:
                        # Use provided 4D mask for current blocks
                        mask_block = mask[:, :, i:min(i+tile_size, seq_len_q), 
                                         j:min(j+tile_size, seq_len_k)]
                    
                    scores = scores.masked_fill(mask_block == 0, float('-inf'))
                
                # Apply softmax and dropout to get attention weights
                attn_weights = F.softmax(scores, dim=-1)
                if self.training and self.dropout > 0:
                    attn_weights = F.dropout(attn_weights, p=self.dropout)
                
                # Compute weighted sum of values
                block_output = torch.matmul(attn_weights, v_block)
                
                # Accumulate results
                acc_att += block_output
                acc_normalizer += attn_weights.sum(dim=-1, keepdim=True)
            
            # Normalize the accumulated attention
            output[:, :, i:min(i+tile_size, seq_len_q), :] = acc_att / (acc_normalizer + 1e-6)
        
        return output
class NSAMoETransformerBlock(nn.Module):
    """
    Transformer block with Natural Sparse Attention and MoE FFN
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_dim: int,
        block_size: int = 16,
        stride: int = 8,
        window_size: int = 128,
        selection_size: int = 128,
        num_experts: int = 8,
        expert_capacity: float = 0.25,
        dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-12,
    ):
        super().__init__()
        
        # Layer norms
        self.pre_attention_norm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.pre_ffn_norm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        
        # Natural Sparse Attention
        self.attention = NaturalSparseAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            block_size=block_size,
            stride=stride,
            window_size=window_size,
            selection_size=selection_size,
            dropout=dropout
        )
        
        # Mixture of Experts FFN
        self.moe = MoELayer(
            hidden_size=hidden_size,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            expert_capacity=expert_capacity,
            dropout=dropout
        )
    
    def forward(self, 
              hidden_states: torch.Tensor, 
              attention_mask: Optional[torch.Tensor] = None,
              past_key_values: Optional[Tuple[torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through transformer block"""
        # Apply attention with residual connection
        norm_hidden_states = self.pre_attention_norm(hidden_states)
        attention_output = self.attention(
            norm_hidden_states, 
            attention_mask=attention_mask,
            past_key_values=past_key_values
        )
        
        hidden_states = hidden_states + attention_output
        
        # Apply MoE with residual connection
        norm_hidden_states = self.pre_ffn_norm(hidden_states)
        moe_output, aux_loss = self.moe(norm_hidden_states)
        
        hidden_states = hidden_states + moe_output
        
        return hidden_states, aux_loss


class NSAMoETransformer(nn.Module):
    """
    Full transformer with Natural Sparse Attention and Mixture-of-Experts FFN
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        block_size: int = 16,
        stride: int = 8,
        window_size: int = 128,
        selection_size: int = 128,
        num_experts: int = 8,
        expert_capacity: float = 0.25,
        max_position_embeddings: int = 2048,
        dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-12
    ):
        super().__init__()
        
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
        # Stack of transformer blocks
        self.layers = nn.ModuleList([
            NSAMoETransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                block_size=block_size,
                stride=stride,
                window_size=window_size,
                selection_size=selection_size,
                num_experts=num_experts,
                expert_capacity=expert_capacity,
                dropout=dropout,
                layer_norm_epsilon=layer_norm_epsilon
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.dropout = nn.Dropout(dropout)
        
        # Output head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Tie embeddings
        self.lm_head.weight = self.embeddings.weight
        
    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
            return_aux_loss: bool = True
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model"""
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        inputs_embeds = self.embeddings(input_ids)
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Get position embeddings
        position_embeds = self.position_embeddings(position_ids)
        
        # Combine token and position embeddings
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=device)
        
        # Initialize auxiliary loss
        total_aux_loss = torch.tensor(0.0, device=device)
        
        # Process layers
        for i, layer in enumerate(self.layers):
            layer_past = None
            if past_key_values is not None:
                layer_past = past_key_values[i]
                
            hidden_states, aux_loss = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=layer_past
            )
            
            total_aux_loss += aux_loss
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        # Project to vocabulary
        logits = self.lm_head(hidden_states)
        
        if return_aux_loss:
            return logits, total_aux_loss
        else:
            return logits, None


# Add the missing train_model function
def train_model(model, dataloader, optimizer, criterion, device, num_epochs=2):
    """Training loop for the transformer model with MPS-specific optimizations"""
    model.train()
    
    # Check if we're using MPS
    is_mps = device.type == 'mps'
    
    # MPS-specific optimizations
    if is_mps:
        # Enable async execution for MPS
        torch._C._set_mps_device_compile(True)
        print("Enabled MPS async compilation")
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_aux_loss = 0
        
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            # Move tensors to device
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Use memory-efficient forward pass
            with torch.cuda.amp.autocast() if device.type == 'cuda' else contextlib.nullcontext():
                logits, aux_loss = model(input_ids)
                
                # Calculate loss
                # Reshape logits for cross entropy
                shifted_logits = logits[:, :-1, :].contiguous()
                shifted_target_ids = target_ids[:, 1:].contiguous()
                
                # Reshape for cross entropy (batch_size * seq_len, vocab_size)
                shifted_logits = shifted_logits.view(-1, shifted_logits.size(-1))
                shifted_target_ids = shifted_target_ids.view(-1)
                
                # Calculate cross entropy loss
                ce_loss = criterion(shifted_logits, shifted_target_ids)
                
                # Combine with auxiliary loss from MoE
                loss = ce_loss
                if aux_loss is not None:
                    loss += aux_loss
                    total_aux_loss += aux_loss.item()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Force synchronization for MPS device
            if is_mps:
                torch.mps.synchronize()
                
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, "
                      f"Loss: {loss.item():.4f}, CE Loss: {ce_loss.item():.4f}, "
                      f"Aux Loss: {aux_loss.item() if aux_loss is not None else 0:.4f}")
        
        avg_loss = total_loss / len(dataloader)
        avg_aux_loss = total_aux_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed, Avg Loss: {avg_loss:.4f}, Avg Aux Loss: {avg_aux_loss:.4f}")
# Main function to run the training
def main():
    try:
        # Set up device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS device (Apple Silicon)")
            # Optimize MPS performance
            torch.backends.mps.enable_operator_fusion()
            import os
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA device")
        else:
            device = torch.device("cpu")
            print("Using CPU device")
        
        # Load the dataset from Hugging Face
        print("Loading dataset...")
        hf_dataset = load_dataset("RedStar-Reasoning/code_dataset", split="train")
        print(f"Loaded dataset with {len(hf_dataset)} examples")
        
        # Initialize the tokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create dataset and dataloader
        seq_len = 128  # Moderate sequence length for faster training
        batch_size = 4  # Adjust based on available memory
        
        dataset = CodeDataset(hf_dataset, tokenizer, seq_len=seq_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print(f"Created dataloader with {len(dataloader)} batches")
        
        # Initialize model with modest size for demonstration
        print("Initializing model...")
        model = NSAMoETransformer(
            vocab_size=tokenizer.vocab_size,
            hidden_size=256,
            num_layers=2,
            num_heads=8,
            ffn_dim=512,
            block_size=16,
            stride=8,
            window_size=64,
            selection_size=64,
            num_experts=4,
            expert_capacity=0.25,
            max_position_embeddings=seq_len,
            dropout=0.1
        )
        model.to(device)
        
        # Print model size
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model initialized with {param_count:,} parameters")
        
        # Initialize optimizer and loss function
        optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        
        # Train model
        print("Starting training...")
        train_model(model, dataloader, optimizer, criterion, device, num_epochs=1)
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
Cell In[47], line 12
     11 try:
---> 12     import sklearn.metrics
     13 except ImportError:

ModuleNotFoundError: No module named 'sklearn.metrics'

During handling of the above exception, another exception occurred:

ModuleNotFoundError                       Traceback (most recent call last)
Cell In[47], line 16
     14     import subprocess
     15     subprocess.check_call(["pip", "install", "scikit-learn", "transformers", "datasets"])
---> 16     import sklearn.metrics
     19 class NaturalSparseAttention(nn.Module):
     20     """
     21     Implementation of Natural Sparse Attention as described in the paper.
     22     """

ModuleNotFoundError: No module named 'sklearn.metrics' - I don't get, I have scikit-learn installed.??/
