import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.decomposition import PCA


def combine_embeddings(price_path, contract_path, output_path, fusion_method='concatenate', **kwargs):
    """
    Combining event embedding vectors and contract embedding vectors.

    Parameter:
    price_path: Event embedding vector CSV file path
    contract_path: Contract embedding vector CSV file path
    output_path: Fused vector output file path
    fusion_method: 'concatenate', 'weighted_concat', 'attention', 'weighted_sum', 'cross_attention', 'gated', 'self_attention_gnn'
    **kwargs: Additional parameters required for the fusion method, such as weights, etc.
    """

    price_df = pd.read_csv(price_path)
    contract_df = pd.read_csv(contract_path)

    if price_df.shape[0] != contract_df.shape[0]:
        raise ValueError("The two files have a different number of lines and cannot be merged.")

    price_vec = price_df.values.astype(np.float32)
    contract_vec = contract_df.values.astype(np.float32)
    fused_vec = None

    # The processing is performed according to the selected fusion method.
    if fusion_method == 'concatenate':
        fused_vec = np.hstack((price_vec, contract_vec))
        columns = [f'price_dim_{i}' for i in range(price_vec.shape[1])] + \
                  [f'contract_dim_{i}' for i in range(contract_vec.shape[1])]

    elif fusion_method == 'weighted_concat':
        event_weight = kwargs.get('event_weight', 0.3)
        code_weight = kwargs.get('code_weight', 0.7)
        fused_vec = weighted_concat_fusion(price_vec, contract_vec, event_weight, code_weight)
        columns = [f'weighted_price_dim_{i}' for i in range(price_vec.shape[1])] + \
                  [f'weighted_contract_dim_{i}' for i in range(contract_vec.shape[1])]

    elif fusion_method == 'attention':
        price_tensor = torch.tensor(price_vec)
        contract_tensor = torch.tensor(contract_vec)
        fusion_model = AttentionFusion(
            event_dim=price_vec.shape[1],
            code_dim=contract_vec.shape[1],
            hidden_dim=kwargs.get('hidden_dim', 512)
        )
        with torch.no_grad():
            fused_tensor = fusion_model(price_tensor, contract_tensor)
        fused_vec = fused_tensor.numpy()
        columns = [f'attention_fused_dim_{i}' for i in range(fused_vec.shape[1])]

    elif fusion_method == 'weighted_sum':
        price_dim = price_vec.shape[1]
        contract_dim = contract_vec.shape[1]
        target_dim = max(price_dim, contract_dim)  # Use the larger dimension as the target dimension.

        # Define the projection layer to ensure dimensional consistency.
        price_proj = nn.Linear(price_dim, target_dim)
        contract_proj = nn.Linear(contract_dim, target_dim)
        fusion_model = WeightedFusion(target_dim)

        # Convert to tensor and process.
        price_tensor = torch.tensor(price_vec)
        contract_tensor = torch.tensor(contract_vec)

        with torch.no_grad():
            price_projected = price_proj(price_tensor)
            contract_projected = contract_proj(contract_tensor)
            fused_tensor = fusion_model(contract_projected, price_projected)

        fused_vec = fused_tensor.numpy()
        columns = [f'weighted_sum_dim_{i}' for i in range(fused_vec.shape[1])]

    elif fusion_method == 'cross_attention':
        price_dim = price_vec.shape[1]
        contract_dim = contract_vec.shape[1]
        target_dim = max(price_dim, contract_dim)

        # Projected onto the same dimension
        price_proj = nn.Linear(price_dim, target_dim)
        contract_proj = nn.Linear(contract_dim, target_dim)
        fusion_model = CrossAttentionFusion(embed_dim=target_dim)


        price_tensor = torch.tensor(price_vec)
        contract_tensor = torch.tensor(contract_vec)

        with torch.no_grad():
            price_projected = price_proj(price_tensor).unsqueeze(1)  # (batch, 1, dim)
            contract_projected = contract_proj(contract_tensor)
            fused_tensor = fusion_model(contract_projected, price_projected)

        fused_vec = fused_tensor.numpy()
        columns = [f'cross_attention_dim_{i}' for i in range(fused_vec.shape[1])]

    elif fusion_method == 'gated':
        price_dim = price_vec.shape[1]
        contract_dim = contract_vec.shape[1]
        target_dim = max(price_dim, contract_dim)

        price_proj = nn.Linear(price_dim, target_dim)
        contract_proj = nn.Linear(contract_dim, target_dim)
        fusion_model = GatedFusion(embed_dim=target_dim)

        price_tensor = torch.tensor(price_vec)
        contract_tensor = torch.tensor(contract_vec)

        with torch.no_grad():
            price_projected = price_proj(price_tensor)
            contract_projected = contract_proj(contract_tensor)
            fused_tensor = fusion_model(contract_projected, price_projected)

        fused_vec = fused_tensor.numpy()
        columns = [f'gated_dim_{i}' for i in range(fused_vec.shape[1])]

    elif fusion_method == 'self_attention_gnn':
        price_dim = price_vec.shape[1]
        contract_dim = contract_vec.shape[1]
        target_dim = max(price_dim, contract_dim) 

        price_proj = nn.Linear(price_dim, target_dim)
        contract_proj = nn.Linear(contract_dim, target_dim)
        fusion_model = SelfAttentionGNNFusion(embed_dim=target_dim)

        price_tensor = torch.tensor(price_vec)
        contract_tensor = torch.tensor(contract_vec)

        with torch.no_grad():
            price_projected = price_proj(price_tensor)
            contract_projected = contract_proj(contract_tensor)
            seq = torch.stack([price_projected, contract_projected], dim=1)
            fused_tensor = fusion_model(seq)

        fused_vec = fused_tensor.numpy()
        columns = [f'self_attn_gnn_dim_{i}' for i in range(fused_vec.shape[1])]

    else:
        raise ValueError(f"Unsupported merging method: {fusion_method}")

    # Generate the output DataFrame and save it.
    combined_df = pd.DataFrame(fused_vec, columns=columns)
    combined_df.to_csv(output_path, index=False)

    print(f"{fusion_method}Fusion complete! Fused vector shape.: {fused_vec.shape}")
    print(f"The file has been saved to: {output_path}")


def weighted_concat_fusion(event_emb, code_emb, event_weight=0.4, code_weight=0.6):
    event_emb_norm = event_emb / (np.linalg.norm(event_emb, axis=1, keepdims=True) + 1e-8)
    code_emb_norm = code_emb / (np.linalg.norm(code_emb, axis=1, keepdims=True) + 1e-8)
    event_emb_weighted = event_emb_norm * event_weight
    code_emb_weighted = code_emb_norm * code_weight
    return np.hstack([event_emb_weighted, code_emb_weighted])

class AttentionFusion(nn.Module):
    def __init__(self, event_dim=256, code_dim=768, hidden_dim=512):
        super().__init__()
        self.event_proj = nn.Linear(event_dim, hidden_dim)
        self.code_proj = nn.Linear(code_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)

    def forward(self, event_emb, code_emb):
        event_proj = self.event_proj(event_emb).unsqueeze(1)
        code_proj = self.code_proj(code_emb).unsqueeze(1)
        concat = torch.cat([event_proj, code_proj], dim=1)
        attn_output, _ = self.attention(concat, concat, concat)
        return torch.mean(attn_output, dim=1)

class WeightedFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, contract_emb, log_emb):
        weights = self.softmax(torch.stack([self.alpha, self.beta]))
        return weights[0] * contract_emb + weights[1] * log_emb

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.contract_proj = nn.Linear(embed_dim, embed_dim)
        self.log_proj = nn.Linear(embed_dim, embed_dim)
        self.fusion_proj = nn.Linear(2 * embed_dim, embed_dim)

    def forward(self, contract_emb, log_emb_seq):
        contract_emb_expand = contract_emb.unsqueeze(1)
        contract_attn_out, _ = self.self_attn(
            query=self.contract_proj(contract_emb_expand),
            key=self.log_proj(log_emb_seq),
            value=self.log_proj(log_emb_seq)
        )
        fused = torch.cat([contract_attn_out.squeeze(1), contract_emb], dim=-1)
        return self.fusion_proj(fused)

class GatedFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Sigmoid()
        )
        self.contract_proj = nn.Linear(embed_dim, embed_dim)
        self.log_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, contract_emb, log_emb):
        concat = torch.cat([contract_emb, log_emb], dim=-1)
        gate = self.gate(concat)
        return gate * self.contract_proj(contract_emb) + (1 - gate) * self.log_proj(log_emb)

class SelfAttentionGNNFusion(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads=4,
                 window_size=10,  # Window size: Only aggregates neighbors within a distance of ≤ window_size.
                 alpha=0.5):  # Attenuation coefficient: The larger the value of α, the faster the weight of neighboring points within the window decays.
        super().__init__()
        # Self-attention layer
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # GNN message passing layer (combining information from the node itself and its neighbors)
        self.message_passing = nn.Linear(2 * embed_dim, embed_dim)
        # Final fused projection layer
        self.fusion_proj = nn.Linear(embed_dim, embed_dim)

        # Window size + attenuation coefficient
        self.window_size = window_size
        self.alpha = alpha
        self.eps = 1e-8  # Avoid division by zero/exponent underflow.

    def forward(self, x):
        # Self-attention processes sequential features.
        attn_output, _ = self.self_attn(x, x, x)  # (batch, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = attn_output.shape
        fused = []

        for i in range(seq_len):
            # Collect neighboring elements within the window (distance ≤ window_size, excluding the element itself).
            neighbors = []  
            distances = []  
            for j in range(seq_len):
                # Filtering criteria: not the same element + distance ≤ window size
                if j != i and abs(i - j) <= self.window_size:
                    neighbors.append(attn_output[:, j, :])
                    distances.append(abs(i - j))  # Record the actual distance (for attenuation calculations)

            # Handling edge cases: no neighbors within the window (e.g., seq_len=1 or boundary nodes without enough neighbors).
            if len(neighbors) == 0:
                neighbor_info = torch.zeros(batch_size, embed_dim, device=attn_output.device)
            else:
                # Calculate the distance-decay weights of neighbors within the window.
                distances_tensor = torch.tensor(distances, device=attn_output.device, dtype=torch.float32)
                # Exponentially decaying weights (higher weights for closer neighbors, lower weights for distant neighbors)
                weights = torch.exp(-self.alpha * distances_tensor)
                # Weight normalization (to prevent scale drift caused by the sum of weights not being equal to 1)
                weights = weights / (weights.sum() + self.eps)

                # Weighted average aggregation of neighbor information within the window.
                neighbors_tensor = torch.stack(neighbors)  # (num_neighbors, batch, embed_dim)
                weighted_neighbors = neighbors_tensor * weights.unsqueeze(1).unsqueeze(2)
                # Weighted summation is used to obtain aggregated neighbor information.
                neighbor_info = weighted_neighbors.sum(dim=0)  # (batch, embed_dim)

            # Combining its own features with the aggregated features of its neighbors.
            node_self_info = attn_output[:, i, :]  
            node_info = torch.cat([node_self_info, neighbor_info], dim=-1)  # (batch, 2*embed_dim)

            # Message passing projection
            fused_node = self.message_passing(node_info)  # (batch, embed_dim)
            fused.append(fused_node)

        # Combine the outputs of all nodes (by averaging them).
        final_fused = torch.stack(fused, dim=1)  # (batch, seq_len, embed_dim)
        final = final_fused.mean(dim=1)  # (batch, embed_dim)

        # Final projection
        return self.fusion_proj(final)


if __name__ == "__main__":
    dataset_name = ['attack incident', 'high_value']
    platform = ['ARB', 'AVAX', 'Base', 'BSC', 'ETH', 'POL']
    model = 'Starcoder_768' # deepseek_768, Qwen_768， Starcoder_768，SmolLM2_768，Codegemma_768
    # List of supported fusion methods
    # fusion_methods = ['concatenate', 'weighted_concat', 'attention', 'weighted_sum', 'cross_attention', 'gated', 'self_attention_gnn']
    fusion_methods = ['weighted_concat']
    error = []

    for dn in dataset_name:
        for pf in platform:
            protocol_path = f"./dataset/{dn}/{pf}/"
            if not os.path.exists(protocol_path):
                continue

            for fusion_method in fusion_methods:
                # Create a separate output directory for each fusion method.
                OUTPUT_PATH = f'./merged_features//{fusion_method}/{model}/'
                if not os.path.exists(OUTPUT_PATH):
                    os.makedirs(OUTPUT_PATH)

                protocol_list = os.listdir(protocol_path)
                for pt in protocol_list:
                    output_file = f"{OUTPUT_PATH}{pt}.csv"
                    if os.path.exists(output_file):
                        print(f"{pt} already exists under the {fusion_method} fusion method.")
                        continue

                    try:
                        if fusion_method == 'weighted_concat':
                            combine_embeddings(
                                price_path=f'./event_feature/embeddings/{dn}/{pf}/{pt}.csv',
                                contract_path=f'./contract_feature/embeddings/{model}/{dn}/{pf}/{pt}.csv',
                                output_path=output_file,
                                fusion_method=fusion_method,
                                event_weight=0.4, 
                                code_weight=0.6
                            )
                        elif fusion_method in ['attention', 'cross_attention']:
                            combine_embeddings(
                                price_path=f'./event_feature/embeddings/{dn}/{pf}/{pt}.csv',
                                contract_path=f'./contract_feature/embeddings/{model}/{dn}/{pf}/{pt}.csv',
                                output_path=output_file,
                                fusion_method=fusion_method,
                                hidden_dim=512
                            )
                        else:
                            combine_embeddings(
                                price_path=f'./event_feature/embeddings/{dn}/{pf}/{pt}.csv',
                                contract_path=f'./contract_feature/embeddings/{model}/{dn}/{pf}/{pt}.csv',
                                output_path=output_file,
                                fusion_method=fusion_method
                            )
                        print(f"{pt}.csv has been saved using the {fusion_method} fusion method.")
                    except Exception as e:
                        error.append(f"An error occurred with the {fusion_method} fusion method: {str(e)}")
                        print(f"An error occurred while saving {pt}.csv using the {fusion_method} fusion method.")

    print("Error List:")
    for err in error:
        print(err)