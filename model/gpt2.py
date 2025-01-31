
import torch
import torch.nn as nn
from torch.nn import functional as F

from attention import CausalSelfAttention

class GPT2Block(nn.Module):
    """GPT-2 Transformer块"""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.n_inner),
            nn.GELU(),
            nn.Linear(config.n_inner, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2(nn.Module):
    """GPT-2模型实现"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # input embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        
        # transformer blocks
        self.blocks = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        
        # final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, position_ids=None):
        b, t = input_ids.size()
        
        # 生成位置编码
        if position_ids is None:
            position_ids = torch.arange(0, t, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        
        # 前向传播
        token_embeddings = self.wte(input_ids)
        position_embeddings = self.wpe(position_ids)
        x = self.drop(token_embeddings + position_embeddings)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        
        # 输出logits
        logits = F.linear(x, self.wte.weight)
        
        return logits

    def generate(self, input_ids, max_length, temperature=1.0, top_k=None):
        """简单的文本生成函数"""
        self.eval()
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # 获取预测
                logits = self(input_ids)
                logits = logits[:, -1, :] / temperature
                
                # top-k采样
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # 采样下一个token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 添加到输入序列
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
        return input_ids

