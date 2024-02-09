from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
#from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import FakeListLLM
from langchain_community.chat_models import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
# from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
#from langchain.callbacks import get_openai_callback
from langchain_community.callbacks import get_openai_callback

import pandas as pd
from pandas import DataFrame
from typing import List
import tiktoken
import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import sys
import pickle
from itertools import chain, tee
from itertools import combinations
import networkx as nx
import json
import random
import choix
import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler
from ranx import Qrels, Run, evaluate
from collections import defaultdict

class AIDTRag:

    SYSTEM_TEMPLATE = """You are a helpful assistant to a Javascript developer. 
    Answer the QUESTION based on the CONTEXT below.     
    If the question cannot be answered using the information provided, simply return an empty list. 
    """

    ZERO_SHOT_TEMPLATE = """CONTEXT: As a JavaScript developer, I want to perform the task indicated
    by the QUESTION below. Return a ranking of {k} suggested JavaScript packages, from best to worst, 
    for the task as a list. The returned packages {environment} environment. {year}
    For each package, include the following data: 
    - package name, 
    - a short description, 
    - its url, 
    - its year of release, 
    - {n} adjectives that justify the package choice, 
    - pros and cons in a concise way; 
    all formatted as JSON objects. 
    Please use underscore and lowercase in the names of the JSON fields.
    Do not make up your answer. 
    If the question cannot be answered using the information provided, simply return an empty list. 

    QUESTION: {task}?
    """

    EXAMPLES_TEMPLATE = """CONTEXT: As a JavaScript developer, I want to perform the task indicated
    by the QUESTION below. Given these JavaScript packages commonly used by developers: {packages},
    rank them in a list of up to {k} packages suitable for the task, from best to worst. 
    The returned packages {environment} environment. {year}
    Use only packages of the provided list, but you might discard packages being not relevant to the task.
    If none of the packages are suitable for the task, return an empty list.
    For each package, include the following data: 
    - package name, 
    - a short description, 
    - its url, 
    - its year of release, 
    - {n} adjectives that justify the package choice, 
    - and pros and cons in a concise way; 
    all formatted as JSON objects. 
    Please use underscore and lowercase in the names of the JSON fields.
    Do not make up your answer. 

    QUESTION: How to {task}?
    """

    EXPLANATION_TEMPLATE = """CONTEXT: As a JavaScript developer, I want to perform the task indicated
    by the QUESTION below. Given these JavaScript packages commonly used by developers: {packages},
    help me to make comparisons among them. 
    The returned packages {environment} environment. {year}
    In your response, include all the packages of the list in the provided order.
    For each package, include the following data: 
    - package name, 
    - a short description, 
    - its url, 
    - its year of release, 
    - {n} adjectives that justify the package choice, 
    - and pros and cons in a concise way; 
    all formatted as JSON objects. 
    Please use underscore and lowercase in the names of the JSON fields.
    Do not make up your answer. 

    QUESTION: How to {task}?
    """

    # EXPLANATION_TEMPLATE = """CONTEXT: As a JavaScript developer, I want to perform the task: {task}. 
    # To do so, I selected the following list of JavaScript packages: {packages},
    # and I need to get additional data about them. 
    # Please answer the QUESTION below and do not make up your answer.

    # QUESTION: For each package in my list, can you provide the following data?: 
    # - package name, 
    # - a short description, 
    # - its url, 
    # - its year of release, 
    # - {n} adjectives that justify the package choice, 
    # - and pros and cons in a concise way; 
    # all formatted as JSON objects?
    # Note that some packages might be unrelated to the task. 
    # Please use underscore and lowercase in the names of the JSON fields.
    # """


    CHROMA_DIRECTORY = "./chroma_db"
    EMBEDDINGS = "paraphrase-MiniLM-L6-v2"
    L2R_MODEL = './models/gbrank.pkl' #'./models/GBRank_models'
    TOP_K = 6
    N_ADJECTIVES = 3
    ENVIRONMENT_CONSTRAINT = "should be compatible with Node.js"
    YEAR_CONSTRAINT = "The packages must have been released before 2018."
    
    @staticmethod
    def load_documents(df, bm25=False):
        #nlp = spacy.load("en_core_web_sm")
        doc_list = []
        for index, row in df.iterrows():
            #print(row['description'])
            d = row['description'].replace('"', '')
            if bm25:
                s = d #s = " ".join(normalize_corpus(d, nlp)) # Clean and tokenize description before ingestion
                doc =  Document(page_content=s, metadata={"source": row['name'], "id": index, "description": d})
            else:
                doc =  Document(page_content=d, metadata={"source": row['name'], "id": index, "justification": ""})
            doc_list.append(doc)
    
        return doc_list


    def __init__(self, dataset: DataFrame, retriever: BaseRetriever =None, ranker=None, k: int=5, llm: BaseChatModel =None) -> None:
        self.TOP_K = k
        self.githubdb_df = dataset #pd.read_csv(dataset)
        #print("Loading dataset ...", dataset, self.githubdb_df.shape)

        if retriever is None:
            if os.path.exists(self.CHROMA_DIRECTORY):
                print("Loading chromadb from disk ...", self.CHROMA_DIRECTORY)
                self.embeddings = SentenceTransformerEmbeddings(model_name=self.EMBEDDINGS)
                db = Chroma(embedding_function=self.embeddings, persist_directory=self.CHROMA_DIRECTORY)
                print(db._collection.count(), "documents")
            else:
                print("Creating chromadb and ingesting documents ...", self.CHROMA_DIRECTORY)
                documents = AIDTRag.load_documents(self.githubdb_df)
                self.embeddings = SentenceTransformerEmbeddings(model_name=self.EMBEDDINGS)
                db = Chroma.from_documents(documents, self.embeddings, persist_directory=self.CHROMA_DIRECTORY)
                print(len(documents), "documents")
            self.retriever = db.as_retriever(search_kwargs={"k": self.TOP_K})
        else:
            print("Using a custom retriever ...", type(retriever))
            retriever.k = k
            self.retriever = retriever
        
        if ranker is None:
            self.classifier = pickle.load(open(self.L2R_MODEL, 'rb')) #pickle.load(open(self.L2R_MODEL, 'rb'))[0]
            print("Loading L2R classifier ...", self.L2R_MODEL)
        else:
            print("Using a custom ranker ...", type(ranker))
            self.classifier = ranker
        
        self.set_llm(None)
    

    def set_reranker(self, compressor):
        compressor.top_n = self.TOP_K
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=self.retriever, 
            ) 
        self.retriever = compression_retriever

    def set_llm(self, llm: BaseChatModel) -> None:
        if llm is None:
            self.llm = FakeListLLM(responses=["Fake response", "Another fake response"])
            self.output_parser =  StrOutputParser() 
        else:
            self.llm = llm
            self.output_parser = SimpleJsonOutputParser()

    # pairwise recipe from the itertools docs.
    @staticmethod
    def pairwise(iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    # https://stackoverflow.com/questions/51429309/logically-sorting-a-list-of-lists-partially-ordered-set-topological-sort
    @staticmethod
    def merge_ordering(sublists):
        # Make an iterator of graph edges for the new graph. Some edges may be repeated.
        # That's fine. NetworkX will ignore duplicates.
        edges = list(chain.from_iterable(map(AIDTRag.pairwise, sublists)))

        graph = nx.DiGraph(edges)
        if not nx.is_directed_acyclic_graph(graph):
            print("Warning: Graph is NOT a DAG!")
            print("Trying to add edges individually to avoid cycles ...")
            graph = nx.DiGraph()
            random.shuffle(edges)
            edges_removed = []
            for e in edges:
                graph.add_edge(*e)
                if not nx.is_directed_acyclic_graph(graph):
                    print("\t skipping edge:", e)
                    graph.remove_edge(*e)     
                    edges_removed.append(e)
            print("edges removed:", len(edges_removed)) 

        return list(nx.algorithms.topological_sort(graph))
    
    @staticmethod
    def choix_ranking(ranking, input_pairs):
        n_items = len(ranking)
        input_pairs_tuples = [(ranking.index(pair[0]), ranking.index(pair[1])) for pair in input_pairs]
        #print(input_pairs_tuples)
        params = choix.ilsr_pairwise(n_items, input_pairs_tuples, alpha=0.01)
        #params = choix.lsr_pairwise(n_items, input_pairs_tuples, alpha=0.01)
        my_order = np.argsort(-params)
        #print("Re-ranking (best-to-worst) - choix:", my_order)
        return [ranking[i] for i in my_order]

    @staticmethod
    def predict_ranking(ranking, df, model, use_choix=False):
        # Computing all combinations (pairs) of the ranking
        ranking_pairs = [pair for pair in combinations(ranking,2)]

        # Converting the pairs to a dataframe (pairwise arrangement)
        first_elem = [pair[0] for pair in ranking_pairs]   
        second_elem = [pair[1] for pair in ranking_pairs]
        first_df = df.loc[first_elem].reset_index().drop("name", axis=1)
        second_df = df.loc[second_elem].reset_index().drop("name", axis=1)
        #first_columns = first_df.columns
        second_columns = [str(col)[:-1]+'2' for col in second_df.columns]
        second_df.columns = second_columns
        concat_df = pd.concat([first_df, second_df], axis=1)
        concat_df = concat_df.fillna(0)
        #print(concat_df.columns)

        # Loading the GBRank classifier and making predictions
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(concat_df)
        results = model.predict(X_scaled)

        # Determine a total order (i.e., topological order) for the results
        input_pairs = []
        for pair, y in zip(ranking_pairs, results):
            if y == 1:
                input_pairs.append(list(pair))
            else:
                input_pairs.append([pair[1], pair[0]])
        
        alternative_ranking =  AIDTRag.choix_ranking(ranking, input_pairs)
        #print("Re-ranking (choix):", alternative_ranking)
        if use_choix:
            return alternative_ranking, results
        else:
            return AIDTRag.merge_ordering(input_pairs), results

    def check_grounding(self, query: str, technologies: List[str]) -> bool:
        results = set(self.retrieve(query))
        llm_technologies = set([t.lower() for t in technologies])
        if len(llm_technologies) < len(technologies):
            print("Warning: LLM returned duplicate technologies")
        common_technologies = results.intersection(llm_technologies)
        if len(common_technologies) == len(results):
            return True
        else:
            diff = llm_technologies.difference(results)
            print("Warning: LLM added", len(diff), "additional technologies", diff)
            return False

#[{'package name': 'quagga', 'description': 'An advanced barcode-reader written in JavaScript', 'url': 'https://serratus.github.io/quaggaJS/', 'year of release': '2014', 'justification': ['advanced', 'popular', 'well-documented'], 'pros': ['Advanced barcode reading capabilities', 'Popular and widely used', 'Well-documented'], 'cons': ['May require additional configuration for specific use cases']}, {'package name': 'dbr', 'description': 'A barcode reading library for Node.js', 'url': 'https://www.dynamsoft.com/Products/Dynamic-Barcode-Reader.aspx', 'year of release': '2013', 'justification': ['specifically designed for Node.js', 'powerful', 'supports various barcode types'], 'pros': ['Specifically designed for Node.js', 'Powerful barcode reading capabilities', 'Supports various barcode types'], 'cons': ['Not as widely used as some other packages']}, {'package name': 'barcode-scanner', 'description': 'A barcode scanner library for Node.js', 'url': 'https://www.npmjs.com/package/barcode-scanner', 'year of release': '2015', 'justification': ['specifically designed for Node.js', 'easy to use', 'supports multiple barcode formats'], 'pros': ['Specifically designed for Node.js', 'Easy to use', 'Supports multiple barcode formats'], 'cons': ['May not have as advanced features as other packages']}, {'package name': 'barcode-js', 'description': 'A JavaScript barcode generator and decoder', 'url': 'https://barcode-js.com/', 'year of release': '2012', 'justification': ['barcode generation and decoding', 'mature', 'supports various barcode types'], 'pros': ['Barcode generation and decoding capabilities', 'Mature package with long history', 'Supports various barcode types'], 'cons': ['May not have as advanced features as other packages']}, {'package name': 'datamatrix-decode', 'description': 'A JavaScript library for decoding Data Matrix barcodes', 'url': 'https://www.npmjs.com/package/datamatrix-decode', 'year of release': '2017', 'justification': ['specifically for Data Matrix barcodes', 'recent', 'lightweight'], 'pros': ['Specifically designed for decoding Data Matrix barcodes', 'Recent package', 'Lightweight'], 'cons': ['Limited to Data Matrix barcodes only']}]
        
    def retrieve(self, query: str, as_json=False) -> List[str] | str: # JSON
        #print(type(self.retriever.vectorstore))
        results = self.retriever.get_relevant_documents(query) #, search_kwargs={"k": k})
        # for doc in results:
        #     print(doc.metadata['source'], doc.metadata['justification']) #doc.page_content)
        if not as_json:
            return [doc.metadata['source'] for doc in results]
        else:
            return_info = []
            for doc in results:
                return_info.append({'package_name': doc.metadata['source'], 
                                    'description': doc.page_content, 
                                    'url': 'null', 
                                    'year_of_release': 'null', 
                                    'justification': doc.metadata['justification'], 
                                    'pros': [], 
                                    'cons': []
                                    })
            return json.dumps(return_info, indent=2)

    
    def rank(self, technologies: List[str]) -> List[str]:
        if len(technologies) == 0:
            return []
        
        features_df = self.githubdb_df.copy()
        features_df.set_index('name', inplace=True)
        features_df.drop('description', axis=1, inplace=True)
        existing_technologies = []
        non_existing_technologies = []
        for tt in technologies:
            if tt in features_df.index:
                existing_technologies.append(tt)
            else:
                #print("Warning: missing features for", tt)
                non_existing_technologies.append(tt)
        if len(non_existing_technologies) > 0:
            print("Warning:", len(non_existing_technologies), "technologies without features, out of", len(technologies))
            print(non_existing_technologies)
        
        re_ranking = []
        if len(existing_technologies) > 1:
            features_df = features_df.loc[existing_technologies]
            #print(features_df)
            re_ranking, _ = AIDTRag.predict_ranking(existing_technologies, features_df, model=self.classifier, use_choix=True)
        elif len(existing_technologies) == 1:
            re_ranking = existing_technologies
        
        return re_ranking + non_existing_technologies
    
    @staticmethod
    def num_tokens_from_string(string: str, encoding_name: str ="cl100k_base") -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
    
    def search(self, query: str, prompt: str=ZERO_SHOT_TEMPLATE) -> str: # JSON
        """Search for the technologies (zero-shot) according to the query"""
        try:
            response = self._generate(query, [], prompt)
            print(response)
        except Exception as e:
            print("Error:", e)
            response = json.dumps([])
        return response
    
    def _generate(self, query: str, technologies: List[str], human_prompt: str) -> str: # JSON

        model_prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_TEMPLATE), # system role
            ("human", human_prompt) # human, the user text   
        ])
        task = query #+" "+context
        message = model_prompt.format(k=self.TOP_K, environment=self.ENVIRONMENT_CONSTRAINT, n=self.N_ADJECTIVES, 
                                      task=task, year=self.YEAR_CONSTRAINT, packages=technologies)
        print("Prompting:", message)
        print(AIDTRag.num_tokens_from_string(message), "tokens (approx.)")

        chain = model_prompt | self.llm | self.output_parser
        response = chain.invoke({"k": self.TOP_K, "environment": self.ENVIRONMENT_CONSTRAINT, "n": self.N_ADJECTIVES, 
                                 "task": task, "year": self.YEAR_CONSTRAINT, "packages": technologies})
        print("Reponse:", response)
        if isinstance(self.llm, FakeListLLM):
            return json.dumps([response])
        else: # It should be a normal LLM (e.g., OpenAI)
            return json.dumps(response)
    
    def _get_descriptions(self, technologies: List[str]) -> List[str]:
        descriptions = []
        all_tecnologies = self.githubdb_df['name'].tolist()
        for t in technologies:
            if t in all_tecnologies:
                d = self.githubdb_df[self.githubdb_df['name'] == t]['description'].values[0]
                descriptions.append(d)
            else:
                descriptions.append(t)
        return descriptions
    
    def execute(self, query: str, rerank: str=None, explain=False) -> str: # JSON
        """The main function to execute the full RAG-LLM pipeline"""
        rerank_option = False
        prompt_option = None
        if rerank is None:
            rerank_option = False
            if explain:
                prompt_option = AIDTRag.EXPLANATION_TEMPLATE
        elif rerank == 'gbrank':
            rerank_option = True
            if explain:
                prompt_option = AIDTRag.EXPLANATION_TEMPLATE
        elif (rerank == 'gpt-3.5') or (rerank == 'cohere'):
            rerank_option = False
            prompt_option = AIDTRag.EXAMPLES_TEMPLATE # This prompt already includes explanations
        else:
            print("Warning: unknown rerank option:", rerank_option, "(and no prompt)")
        
        try:
            response = self._do_rag(query, rerank=rerank_option, prompt=prompt_option)
        except Exception as e:
            print("Error:", e)
            response = json.dumps([])
        return response
    
    def _do_rag(self, query: str, rerank: bool=False, prompt: str=None) -> str: # JSON
        """Internal execution of the full RAG-LLM pipeline. The prompt might be customized"""
        technologies = self.retrieve(query)
        if len(technologies) == 0:
            return json.dumps(technologies)
        
        if rerank:
            technologies = self.rank(technologies)
            #print(technologies)

        if prompt is not None: # Run the LLM with this prompt (as a chat model)
            return self._generate(query, technologies, prompt)
        else:
            return_info = []
            descriptions = self._get_descriptions(technologies)
            for t,d in zip(technologies, descriptions):
                return_info.append({'package_name': t, 
                                    'description': d, 
                                    'url': 'null', 
                                    'year_of_release': 'null', 
                                    'justification': [], 
                                    'pros': [], 
                                    'cons': []
                                    })
            return json.dumps(return_info, indent=None)


class STRetriever(BaseRetriever):

    query_dict: dict = {}
    docs_dict: dict = {}
    k: int = 3

    @staticmethod
    def _parse_st_retrieval_file(df: DataFrame, col='borda_fuse') -> dict:
        queries = dict()
        for index, row in df.iterrows():
            #print(index, row['borda_fuse'])
            queries[row['query']] = row[col] #eval(row[col]) # It comes originally as a string list
        #print(queries)
        return queries
    
    @staticmethod
    def from_documents(documents: List[Document], df: DataFrame, col='borda_fuse') -> BaseRetriever:
        retriever = STRetriever()
        retriever._load_results(df, documents, col=col)
        return retriever

    def _load_results(self, df: DataFrame, documents: List[Document], col='borda_fuse') -> None:
        self.query_dict = STRetriever._parse_st_retrieval_file(df, col=col)
        #print(documents)
        for query in self.query_dict.keys():
            for doc in documents:
                if doc.metadata['source'] in self.query_dict[query]:
                    self.docs_dict[doc.metadata['source']] = doc
        print("Documents matched:", len(self.docs_dict))
        #print(self.docs_dict.keys())

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        list_docs = []
        if query in self.query_dict.keys():
            docs = self.query_dict[query]
            n = self.k
            if len(docs) > self.k:
                n = self.k
            for name in self.query_dict[query][0:n]:
                if name in self.docs_dict.keys():
                    list_docs.append(self.docs_dict[name])
                else:
                    list_docs.append(Document(page_content=name, metadata={"source": name, "id": -1, "justification": []}))
        #print("STRetrieval:", list_docs)
        return list_docs
    
    def add_justifications(self, justifications: dict) -> None:
        for k in justifications.keys():
            if k in self.docs_dict.keys():
                self.docs_dict[k].metadata['justification'] = justifications[k]
    

# TODO: Implement here the metrics for ranking evaluation (e.g., using ranx)
class AIDTEvaluator:

    METRICS: List[str] = ['map@3', 'mrr@3','ndcg@3','precision@3','recall@3','f1@3',
                          'map@5', 'mrr@5','ndcg@5','precision@5','recall@5','f1@5',
                          'map@7', 'mrr@7','ndcg@7','precision@7','recall@7','f1@7'
                          ]

    def __init__(self, queries_hits: DataFrame, q: int=0) -> None:
        self.ground_truth = {} # construimos el ground truth, todos tienen el mismo valor (no hay ranking)
        q_col = queries_hits.columns[q]
        for i in range(0,len(queries_hits)):
            if len(queries_hits['hits'].values[i]) > 0:
                #self.ground_truth[queries_hits.index[i]] = {x:1 for x in queries_hits['hits'].values[i]}
                self.ground_truth[queries_hits.at[i,q_col]] = {x:1 for x in queries_hits['hits'].values[i]}
            else:
                print('-- Warning: skipping', queries_hits.at[i,q_col], "/ 0 hits")


    def get_metrics(self, query_rankings_df: DataFrame) -> defaultdict:

        df = query_rankings_df
        column = df.columns[1]
        results = defaultdict(defaultdict(list).copy) # {query : {metric : list}}

        for i in tqdm(range(0,len(df)), "queries"): # por cada uno hay que hacer el run porque no se pueden combinar múltiples
            # habría que agregar a alguna lista o algo
            query = df.at[i, 'query'] # df.index[i]
            if query not in self.ground_truth:
                print('-- Warning: not in ground truth:', df.index[i], query)
                continue

            rankings = df[column].values[i]
        
            if len(rankings) == 0:
                for k in self.METRICS:
                    results[query][k].append(0)
                continue
        
            if isinstance(rankings[0],list):
                for r in rankings:
                    if isinstance(r,float):
                        continue
                    if len(r) == 0:
                        continue
                
                    run_dict = {}
                    run_dict[query] = {r[i]:len(r)-i for i in range(0,min(len(r),10))}
                    run = Run(run_dict)
                    rr = evaluate(Qrels({query : self.ground_truth[query]}), run, self.METRICS, make_comparable=True, return_mean=False)
                    for k,v in rr.items():
                        results[query][k].append(v[0])
            else:            
                run_dict = {query : {rankings[i]:len(rankings)-i for i in range(0,min(len(rankings),10))}}
                run = Run(run_dict)
                rr = evaluate(Qrels({query : self.ground_truth[query]}), run, self.METRICS, make_comparable=True, return_mean=False)
                for k,v in rr.items():
                    results[query][k].append(v[0])
        
            if query not in results:
                for k in self.METRICS:
                    results[query][k].append(0)

        return results
    
    @staticmethod
    def get_metrics_by_type(results: defaultdict) -> defaultdict:
        metrics_dict = defaultdict(list).copy()
        for m in AIDTEvaluator.METRICS:
            for k in results.keys():
                metrics_dict[m].extend(results[k][m])
        return metrics_dict

    @staticmethod
    def get_metrics_as_dataframe(results: defaultdict, who: str=None) -> DataFrame:
        results_metrics = []
        for q,v in results.items():
            for k,m in v.items():
                results_metrics.extend([{'query':q,'metric': k,'value':mm, 'who':'humanito'} for mm in m])
        df = pd.DataFrame(results_metrics)
        if who is not None:
            df['who'] = who
        return df
    
    
    


