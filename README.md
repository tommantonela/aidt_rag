# AIDT RAG
Reproducibility kit for the paper *The JavaScript Package Selection Task: A Comparative Experiment Using an LLM-based Approach*.

After configuring an appropriate Python environment (e.g., using conda), you can run:
* a standalone version of the RAG system
* the experiments reported in the paper

For a quick start with the RAG system, simply run:
```python 
import pandas as pd
from io import StringIO
from langchain_openai import ChatOpenAI
from rag import AIDTRag

# Configuring OpenAI (GPT)
OPENAI_API_KEY = "your OpenAI API KEY goes here"

# Database of technologies
GITHUB_DATASET = "./data/libraries_github.pkl"
technologies_df = pd.read_pickle(GITHUB_DATASET) 
# Ingestion of the technologies as documents for the database
documents = AIDTRag.load_documents(technologies_df)

TOP_K = 5
rag = AIDTRag(technologies_df, k=TOP_K)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
rag.set_llm(llm)

query = "extract a barcode from an image"
print("Query:", query)
print()

print("Zero-shot:", TOP_K)
# Invoke zero-shot strategy
zeroshot_json = rag.search(query)
print(zeroshot_json)

print("Retrieval + GPT-3.5 (RAG):", TOP_K)
# Invoke RAG strategy
rag_json = rag.execute(query, rerank='gpt-3.5', explain=True)
print(rag_json)

print()
# ranking_df = pd.read_json(StringIO(zeroshot_json))
ranking_df = pd.read_json(StringIO(rag_json))
ranking_df.head()
```
