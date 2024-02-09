import pandas as pd
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.chat_models import ChatOllama
import os
import pickle

from rag import AIDTRag, STRetriever

# In this experiment, we use Llama2 (via Ollama) as a search engine (in the wild) that tries to retrieve
# the different packages for a given need (i.e., query)
# ---------------------------------------------------------------------------------------------

# Inputs
GROUND_TRUTH = "./data/queries_x_hits.csv"
GITHUB_DATASET = "./data/libraries_github.pkl"
QUERIES = "./inputs/queries.csv"
ST_RETRIEVAL_RANKINGS = "./data/bordafuse_rankings.csv"
# Outputs
#OUTPUT_RANKINGS = "./data/rankings_experiment4b.csv"
#OUTPUT_METRICS = "./data/metrics_experiment4b.csv"
JSON_OUTPUT_FILE = "./data/json_experiment4b.json"

TOP_K = 10
selector = 'borda_fuse' # 'humans' # 'npmjs' # 'borda_fuse'

# Database of technologies
technologies_df = pd.read_pickle(GITHUB_DATASET) 
#df['description'] = df['description'].fillna(value="")
rankings_df = pd.read_csv(ST_RETRIEVAL_RANKINGS)
rankings_df[selector] = rankings_df[selector].apply(eval) # It is a string, so we need to convert it to a list
rankings_df.sort_values('query', inplace=True)

# Ingestion of the technologies as documents for the database
documents = AIDTRag.load_documents(technologies_df)

# Configuration of a fake retriever with the human rankings (based on the AIDT tool)
st_retriever = STRetriever.from_documents(documents, rankings_df, col=selector)

my_model = "llama2:7b-chat" #"mistral:7b-instruct"
rag = AIDTRag(technologies_df, k=TOP_K, retriever=st_retriever)
llm = ChatOllama(
    model=my_model, temperature=0.0, format="json",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)
rag.set_llm(llm)

selector = 'ollama-'+my_model
print(selector)
# These are the reference queries for the experiment
userstories_df = pd.read_csv(QUERIES)
#userstories_df = userstories_df.head(3)
print() 
# Run the RAG for all the queries
python_objects = []
temporal_file = "./temporal_prompts.pkl"
if temporal_file is not None:
    if os.path.exists(temporal_file):
        with open(temporal_file, 'rb') as input:
            temporal_dict = pickle.load(input)
    else:
        temporal_dict = dict()
else:
    temporal_dict = dict()
print(len(temporal_dict.keys()), "queries recovered")

for index, row in userstories_df.iterrows():
    print(index, "-"*100)
    query = row['query']
    if query not in temporal_dict:
        print("Query:", query)
        # Only retrieval is needed in this experiment
        reranking_json = rag.execute(query, rerank='cohere', explain=True)
        print("Retrieval + cohere:", TOP_K)
        temporal_dict[query] = reranking_json
        print()
        if temporal_file is not None:
            with open(temporal_file, 'wb') as output:
                pickle.dump(temporal_dict, output)

print("-"*100)

python_objects = [obj_json for obj_json in temporal_dict.values()]
print(len(python_objects), "queries processed")
print()
with open(JSON_OUTPUT_FILE, "w") as f:
     f.write("["+",\n".join(python_objects)+"]")
print()
