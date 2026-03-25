import os
import re
import time
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GemmaTokenizer

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  
os.environ["TRANSFORMERS_VERBOSITY"] = "info"  
snapshot_download.__defaults__ = (None, None, None, None, None, None, True, None, None, tqdm)  

contract_emb = {} # Key: Contract source code, Value: Embedding

print(f"Is the GPU available? {torch.cuda.is_available()}")
print(f"CUDA version：{torch.version.cuda}")  
print(f"torch version：{torch.__version__}")  

torch.cuda.empty_cache()
torch.set_num_threads(1)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    # torch.cuda.set_per_process_memory_fraction(0.8)

# 4-bit quantization core configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  
    bnb_4bit_use_double_quant=True,  
    bnb_4bit_quant_type="nf4",  
    # bnb_4bit_compute_dtype=torch.float16,  
    bnb_4bit_compute_dtype=torch.bfloat16, 
    bnb_4bit_quant_storage_dtype=torch.uint8, 
)

# "deepseek-ai/deepseek-coder-1.3b-instruct" 
# "Qwen/Qwen2.5-Coder-1.5B" 
# "bigcode/starcoderbase-1b" 
# "HuggingFaceTB/SmolLM2-1.7B-Instruct" 
# "google/codegemma-2b" 

model_name = "" # Write your model
print(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id  # Explicitly setting the ID prevents errors during encoding.
print("pad_token：", tokenizer.pad_token)  
print("pad_token_id：", tokenizer.pad_token_id)  

st_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    local_files_only=True,   
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    # offload_folder="./offload",  # Offload to disk when there is insufficient video memory.
)

# Freeze all parameters.
st_model.eval()  
for param in st_model.parameters():
    param.requires_grad = False 



def extract_number_from_filename(filename):
    """
    Extract the numbers at the beginning of the filename.
    Returns: The extracted numbers.
    """
    match = re.match(r"^(\d+\.?\d*)", filename)
    if match:
        return float(match.group(1))  # Convert to floating-point numbers to avoid integer/decimal sorting issues.
    return 0 



# Define the global dimensionality reduction layer.
dim_reducer = None  # Global variables, to avoid repeated initialization.

@torch.no_grad()
def get_embeddings(texts, batch_size=1, max_length=2048, target_dim=768, is_dim_reduce=True):
    """
    The embedding of the specified dimension is obtained by averaging the last layer's hidden states and then applying linear dimensionality reduction.
    :param target_dim: Target dimension
    """
    global dim_reducer
    device = next(st_model.parameters()).device
    all_embs = []
    # Native input dimension
    input_dim = st_model.config.hidden_size

    # Initialize the linear dimensionality reduction layer.
    if dim_reducer is None:
        dim_reducer = nn.Linear(input_dim, target_dim, dtype=torch.bfloat16).to(device)
        dim_reducer.eval() 

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch_texts = texts[i: i + batch_size]
        for text in batch_texts:  
            # Check for empty text.
            if not text.strip(): 
                if is_dim_reduce == True:
                    zero_emb = [0.0] * target_dim 
                else:
                    zero_emb = [0.0] * input_dim 
                all_embs.append(zero_emb)
                continue

            if text in contract_emb:
                all_embs.append(contract_emb[text])
                continue

            # Single text encoding (to avoid interference from the batch dimension)
            enc = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            # Check the length of the encoded sequence.
            if enc.input_ids.size(1) == 0:
                if is_dim_reduce == True:
                    zero_emb = [0.0] * target_dim 
                else:
                    zero_emb = [0.0] * input_dim 
                all_embs.append(zero_emb)
                continue

            outputs = st_model(**enc, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]

            # Average pooling
            attention_mask = enc["attention_mask"].unsqueeze(-1)
            masked = last_hidden * attention_mask
            lengths = attention_mask.sum(dim=1)
            mean_pooled = masked.sum(dim=1) / lengths

            # Dimensionality reduction + normalization
            if is_dim_reduce == True:
                mean_pooled = dim_reducer(mean_pooled) 
            mean_pooled = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

            # Convert to a one-dimensional list
            emb = mean_pooled.cpu().tolist()[0]
            contract_emb[text] = emb
            all_embs.append(emb)

    return all_embs



def main():
    datasets = ['attack incident', 'high_value'] # attack incident, 'high_value'
    # datasets = ['high_value_full']
    platform = ['ARB', 'AVAX', 'Base', 'BSC', 'ETH', 'POL']

    for d in datasets:
        for pf in platform:
            if os.path.exists('../dataset/' + d + '/' + pf + '/'):
                protocol_dir = os.listdir('../dataset/' + d + '/' + pf + '/')
                for pt in protocol_dir:

                    start = time.time()

                    if os.path.exists('./embeddings/' + d + '/' + pf + '/' + pt + '.csv'):
                        print(pt + "The embedding already exists.")
                        continue

                    contract_folder = '../dataset/' + d + '/' + pf + '/' + pt + '/source/'
                    contract_texts = []
                    contract_names = []

                    # First, collect the paths and names of all the .sol files.
                    sol_files = []
                    for filename in os.listdir(contract_folder):
                        if filename.endswith(".sol"):
                            sol_files.append(filename)

                    # Sort the files according to the extracted numbers.
                    sol_files_sorted = sorted(sol_files, key=extract_number_from_filename)

                    # Read the contents of the sorted file.
                    for filename in sol_files_sorted:
                        file_path = os.path.join(contract_folder, filename)
                        with open(file_path, "r", encoding="utf-8") as f:
                            # Read the complete contract source code.
                            contract_text = f.read()
                            # contract_text = f.read()[:2000]  # Limit the read length.
                            contract_texts.append(contract_text)
                            contract_names.append(filename)

                    embeddings = get_embeddings(contract_texts, batch_size=1, max_length=1024, target_dim=768, is_dim_reduce=True)

                    embedding_columns = [f'embedding_dim_{i}' for i in range(len(embeddings[0]))]
                    embedding_df = pd.DataFrame(embeddings, columns=embedding_columns)

                    OUTPUT_PATH = './embeddings/'+ d + '/' + pf + '/'

                    if not os.path.exists(OUTPUT_PATH):
                        os.makedirs(OUTPUT_PATH)
                    embedding_df.to_csv(OUTPUT_PATH + pt + ".csv", index=False)
                    print(pt + " The embedding has been saved.")

                    end = time.time()
                    print('get time: ' + pt + '   ' + str(end - start))

if __name__ == "__main__":
    main()