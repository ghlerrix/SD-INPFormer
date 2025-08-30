import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedMemoryMatrix(nn.Module):
    def __init__(self, memory_slots, key_size, init_temperature=0.1):
        super().__init__()
        self.memory_slots = memory_slots
        self.key_size = key_size
        
        self.keys = nn.Parameter(torch.randn(memory_slots, key_size))
        self.values = nn.Parameter(torch.zeros(memory_slots, key_size))
        
        self.write_heads = nn.Sequential(
            nn.Linear(key_size, 128),
            nn.GELU(),
            nn.Linear(128, memory_slots)
        )
        
        self.temperature = nn.Parameter(torch.tensor([init_temperature]))
        
        nn.init.xavier_uniform_(self.keys)
        self.values.data.zero_()

    def _synchronize_device(self, tensor):
        if self.keys.device != tensor.device:
            return tensor.to(self.keys.device)
        return tensor

    def read(self, query):

        query = self._synchronize_device(query)
        
        orig_shape = query.shape
        query_flat = query.view(-1, self.key_size)  # [batch*seq, key_size]
        
        attn_logits = torch.matmul(query_flat, self.keys.T) / self.temperature.clamp(min=1e-6)
        attn_weights = F.softmax(attn_logits, dim=-1)  # [batch*seq, memory_slots]
        
        mem_read = torch.matmul(attn_weights, self.values)  # [batch*seq, key_size]
        return mem_read.view(*orig_shape)  

    def write(self, query, strength=0.1):

        query = self._synchronize_device(query)
        
        query_flat = query.view(-1, self.key_size)  # [N, key_size], N = batch*seq
        
        write_weights = torch.sigmoid(self.write_heads(query_flat))  # [N, memory_slots]
        
        update_matrix = torch.matmul(
            write_weights.t(),   # [memory_slots, N]
            query_flat           # [N, key_size]
        )  # 结果: [memory_slots, key_size]
        
        self.values.data += strength * update_matrix

class MemoryLayerWithLNResidual(nn.Module):
    def __init__(self, input_dim, output_dim, memory_slots=128, 
                 use_residual=True, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.use_residual = use_residual
        
        if memory_slots == "auto":
            memory_slots = max(32, min(512, input_dim // 4))
        self.memory = EnhancedMemoryMatrix(memory_slots, input_dim)
        
        self.proj_in = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU()
        ) if input_dim != output_dim else nn.Identity()
        
        self.memory_fusion = nn.Linear(input_dim * 2, input_dim)
        
        self.proj_out = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.Dropout(dropout)
        )
        
        self.post_norm = nn.LayerNorm(output_dim) if use_residual else nn.Identity()

    def forward(self, x):

        identity = x
        
        x = self.proj_in(x)
        
        if self.memory.keys.device != x.device:
            self.memory.to(x.device)
        
        mem_read = self.memory.read(x)  
        
        self.memory.write(x)
        
        fused = torch.cat([x, mem_read], dim=-1)
        fused = self.memory_fusion(fused)
        
        out = self.proj_out(fused)
        
        if self.use_residual:

            identity = self._synchronize_device(identity)
            
            if identity.shape[-1] != out.shape[-1]:

                if not hasattr(self, 'identity_proj'):
                    self.identity_proj = nn.Linear(identity.shape[-1], out.shape[-1]).to(out.device)
                identity = self.identity_proj(identity)
            
            out = out + identity
            out = self.post_norm(out)
            
        return out

    def _synchronize_device(self, tensor):

        if self.proj_out[1].weight.device != tensor.device:
            return tensor.to(self.proj_out[1].weight.device)
        return tensor