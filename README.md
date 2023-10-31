# ember-v1

<p align="center">
<img src="https://console.llmrails.com/assets/img/logo-black.svg" width="150px">
</p>

This model has been trained on an extensive corpus of text pairs that encompass a broad spectrum of domains, including finance, science, medicine, law, and various others. During the training process, we incorporated techniques derived from the [RetroMAE](https://arxiv.org/abs/2205.12035) and [SetFit](https://arxiv.org/abs/2209.11055) research papers.

We are pleased to offer this model as an API service through our platform, [LLMRails](https://llmrails.com/?ref=ember-v1). If you are interested, please don't hesitate to sign up.

### Plans
- The research paper will be published soon.
-  The v2 of the model is currently in development and will feature an extended maximum sequence length of 4,000 tokens.

## Usage
Use with API request:
```bash
curl --location 'https://api.llmrails.com/v1/embeddings' \
--header 'X-API-KEY: {token}' \
--header 'Content-Type: application/json' \
--data '{
   "input": ["This is an example sentence"],
   "model":"embedding-english-v1" # equals to ember-v1
}'
```
API docs: https://docs.llmrails.com/embedding/embed-text<br>
Langchain plugin: https://python.langchain.com/docs/integrations/text_embedding/llm_rails

Use with transformers:
```python
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

input_texts = [
    "This is an example sentence",
    "Each sentence is converted"
]

tokenizer = AutoTokenizer.from_pretrained("llmrails/ember-v1")
model = AutoModel.from_pretrained("llmrails/ember-v1")

# Tokenize the input texts
batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

outputs = model(**batch_dict)
embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# (Optionally) normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings[:1] @ embeddings[1:].T) * 100
print(scores.tolist())
```

Use with sentence-transformers:
```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

sentences = [
	"This is an example sentence",
    "Each sentence is converted"
]

model = SentenceTransformer('llmrails/ember-v1')
embeddings = model.encode(sentences)
print(cos_sim(embeddings[0], embeddings[1]))
```

## Massive Text Embedding Benchmark (MTEB) Evaluation
Our model achieve state-of-the-art performance on [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

|                               Model Name                                | Dimension | Sequence Length | Average (56) | 
|:-----------------------------------------------------------------------:|:---------:|:---:|:------------:|
| [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) |   1024    |       512       |    64.23     |  
| [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) |   768    |       512       |    63.55     |
| [ember-v1](https://huggingface.co/llmrails/emmbedding-en-v1) |   1024    | 512 |    **63.54**     |  
| [text-embedding-ada-002](https://platform.openai.com/docs/guides/embeddings/types-of-embedding-models) |   1536    |      8191       |    60.99     |

### Limitation

This model exclusively caters to English texts, and any lengthy texts will be truncated to a maximum of 512 tokens.