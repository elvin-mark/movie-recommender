import pandas as pd
import torch.nn.functional as F
from torch import Tensor
import torch
from transformers import AutoTokenizer, AutoModel
import tqdm
import pickle


dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


tokenizer = AutoTokenizer.from_pretrained("Supabase/gte-small")
model = AutoModel.from_pretrained("Supabase/gte-small").to(dev)

df = pd.read_csv("./assets/movies_metadata.csv")

# Dropping all rows that do not have overview
df_clean = df[["id", "overview"]].dropna()


def embed(i, j):
    input_texts = df_clean["overview"].iloc[i:j].to_list()
    batch_dict = tokenizer(
        input_texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
    ).to(dev)
    with torch.no_grad():
        outputs = model(**batch_dict)
        embeddings = average_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu()


batch = 1000
res = []
for i in tqdm.tqdm(range(len(df_clean) // batch + 1)):
    res.append(embed(i * batch, (i + 1) * batch))

all_embeddings = torch.cat(res)

with open("./movies_embeddings.pkl", "wb") as f:
    pickle.dump(all_embeddings.numpy(), f)
