# Arquiteturas GPT
import torch
import torch.nn as nn

# Transformer Block
class Transformer(nn.Module):
    def __init__(self, N_embd, N_Layers, N_head, drop0):
        super().__init__()
        self.N_Layers = N_Layers
        self.attn_weights = []
        self.MH = nn.ModuleList([
            nn.MultiheadAttention(N_embd, N_head, dropout=drop0, batch_first=True)
            for _ in range(N_Layers)
        ])
        self.scale1 = nn.ModuleList([nn.LayerNorm(N_embd) for _ in range(N_Layers)])
        self.scale2 = nn.ModuleList([nn.LayerNorm(N_embd) for _ in range(N_Layers)])
        self.FFN = nn.ModuleList([
            nn.Sequential(
                nn.Linear(N_embd, 4 * N_embd),
                nn.GELU(),
                nn.Linear(4 * N_embd, N_embd),
                nn.Dropout(drop0)
            ) for _ in range(N_Layers)
        ])

    def forward(self, output, causal_mask):
        self.attn_weights = []
        for j in range(self.N_Layers):
            Q = self.scale1[j](output)
            attn_out, attn_w = self.MH[j](
                Q, Q, Q,
                attn_mask=causal_mask,
                average_attn_weights=False
            )
            self.attn_weights.append(attn_w)
            output = output + attn_out
            norm_out = self.scale2[j](output)
            output = output + self.FFN[j](norm_out)
        
        return output, self.attn_weights


# GPT
class GPT(nn.Module):
    def __init__(self, N_block, N_embd, N_Layers, N_voc, N_head, drop0):
        super().__init__()
        self.wte = nn.Embedding(N_voc, N_embd, padding_idx=0)
        self.wpe = nn.Embedding(N_block, N_embd)
        self.drop0 = nn.Dropout(drop0)
        self.TF = Transformer(N_embd, N_Layers, N_head, drop0)
        self.L1 = nn.Linear(N_embd, N_voc)

    def forward(self, x):
        B, N = x.size()
        pos = torch.arange(0, N, dtype=torch.long, device=x.device).unsqueeze(0)  # [1, N]
        
        causal_mask = torch.full((N, N), float('-inf'), device=x.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        
        output = self.wte(x) + self.wpe(pos)                  # [B, N, N_embd]
        output = self.drop0(output)                           # [B, N, N_embd]
        output = self.TF(output, causal_mask)[0]              # [B, N, N_embd]
        output = self.L1(output)                              # [B, N, N_voc]
        return output

#x = torch.randint(1, 60, size=(5, 64))
#model = GPT(N_block=64, N_embd=128, N_Layers=2, N_voc=60, N_head=2, drop0=0.1)
#total_params = sum(p.numel() for p in model.parameters())
#print(f"Total de parâmetros: {total_params}")
#fit = model(x)
#fit.shape
