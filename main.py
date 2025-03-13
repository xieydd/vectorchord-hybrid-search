from loader import GenericDataLoader
from evaluate import EvaluateRetrieval
from embedding import SentenceEmbedding
from FlagEmbedding import FlagReranker

import argparse
import os
import logging
import json
from time import perf_counter
import zipfile
import requests
from tqdm.autonotebook import tqdm
import psycopg2
from pgvector.psycopg2 import register_vector
from dataclasses import dataclass


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(process)d - %(levelname)s - %(filename)s:%(lineno)s - %(message)s"
)
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)

BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"


def download_and_unzip(url: str, out_dir: str, chunk_size: int = 1024) -> str:
    os.makedirs(out_dir, exist_ok=True)
    dataset = url.split("/")[-1]
    zip_file = os.path.join(out_dir, dataset)

    if not os.path.isfile(zip_file):
        logger.info("Downloading {} ...".format(dataset))
        download_url(url, zip_file, chunk_size)

    if not os.path.isdir(zip_file.replace(".zip", "")):
        logger.info("Unzipping {} ...".format(dataset))
        unzip(zip_file, out_dir)

    return os.path.join(out_dir, dataset.replace(".zip", ""))


def unzip(zip_file: str, out_dir: str):
    zip_ = zipfile.ZipFile(zip_file, "r")
    zip_.extractall(path=out_dir)
    zip_.close()


def download_url(url: str, save_path: str, chunk_size: int = 1024):
    """Download url with progress bar using tqdm
    https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads

    Args:
        url (str): downloadable url
        save_path (str): local path to save the downloaded file
        chunk_size (int, optional): chunking of files. Defaults to 1024.
    """
    r = requests.get(url, stream=True)
    total = int(r.headers.get("Content-Length", 0))
    with (
        open(save_path, "wb") as fd,
        tqdm(
            desc=save_path,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=chunk_size,
        ) as bar,
    ):
        for data in r.iter_content(chunk_size=chunk_size):
            size = fd.write(data)
            bar.update(size)


@dataclass
class DBConfig:
    dbname: str
    user: str
    password: str
    host: str
    port: str

    def get_connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset", default="fiqa", choices=["fiqa", "msmarco", "quora","scidocs","scifact","trec-covid"]
    )
    parser.add_argument("-k", "--topk", default=10, type=int)
    parser.add_argument("-s", "--save_dir", default="datasets")
    # PostgreSQL configuration parameters
    parser.add_argument("--dbname", default="postgres", help="Database name")
    parser.add_argument("--user", default="postgres", help="Database user")
    parser.add_argument("--password", default="postgres", help="Database password")
    parser.add_argument("--host", default="0.0.0.0", help="Database host")
    parser.add_argument("--port", default="5432", help="Database port")
    parser.add_argument("--vector-dim", type=int, default=1024, help="Vector dimension")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of documents to process")
    parser.add_argument("--only-vector", action="store_true", help="Only use vector search")
    parser.add_argument("--only-bm25", action="store_true", help="Only use bm25 search")
    parser.add_argument("--distance", default="cosine", choices=["cosine", "euclidean", "dot"])
    return parser


class PgClient:
    def __init__(self, db_config: DBConfig, dataset: str, num: int, vector_dim: int):
        self.dataset = dataset
        self.num = num
        self.vector_dim = vector_dim
        self.conn = psycopg2.connect(db_config.get_connection_string())
        self.conn.autocommit = True  # Set autocommit mode
        with self.conn.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vchord CASCADE;")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vchord_bm25 CASCADE;")
            cursor.execute('ALTER SYSTEM SET search_path TO "$user", public, bm25_catalog;')
            cursor.execute("SELECT pg_reload_conf();")
        register_vector(self.conn)
        self.sentence_encoder = SentenceEmbedding()

    def create(self):
        with self.conn.cursor() as cursor:
            cursor.execute(f"DROP TABLE IF EXISTS {self.dataset}_corpus;")
            cursor.execute(
                f"CREATE TABLE IF NOT EXISTS {self.dataset}_corpus (id TEXT, text TEXT, emb vector({self.vector_dim}), bm25 bm25vector);"
            )
            cursor.execute(f"DROP TABLE IF EXISTS {self.dataset}_query;")
            cursor.execute(
                f"CREATE TABLE IF NOT EXISTS {self.dataset}_query (id TEXT, text TEXT, emb vector({self.vector_dim}), bm25 bm25vector);"
            )

    def insert(self, doc_ids, docs, qids, queries):
        start_time = perf_counter()

        with self.conn.cursor() as cursor:
            for did, doc in tqdm(zip(doc_ids, docs), desc="insert corpus"):
                emb = self.sentence_encoder.encode_doc(doc)
                cursor.execute(
                    f"INSERT INTO {self.dataset}_corpus (id, text, emb) VALUES (%s, %s, %s)",
                    (did, doc, emb),
                )

            for qid, query in tqdm(zip(qids, queries), desc="insert query"):
                emb = self.sentence_encoder.encode_query(query)
                cursor.execute(
                    f"INSERT INTO {self.dataset}_query (id, text, emb) VALUES (%s, %s, %s)",
                    (qid, query, emb),
                )
            
            query = f"""SELECT create_tokenizer('{self.dataset}_token', $$
                tokenizer = 'unicode'
                stopwords = 'nltk'
                table = '{self.dataset}_corpus'
                column = 'text'
                $$);"""
            cursor.execute(query)
            cursor.execute(
                f"UPDATE {self.dataset}_corpus SET bm25 = tokenize(text, '{self.dataset}_token')"
            )

        logger.info(
            "insert %s in %f seconds", self.dataset, perf_counter() - start_time
        )

    def index(self, workers: int, distance: str):
        start_time = perf_counter()
        centroids = min(4 * int(self.num**0.5), self.num // 40)
        
        if distance == "euclidean":
            ops = "vector_l2_ops"
            spherical_centroids = "false"
        elif distance == "cosine":
            ops = "vector_cosine_ops"
            spherical_centroids = "true"
        else:
            ops = "vector_ip_ops"
            spherical_centroids = "true"
        ivf_config = f"""
        residual_quantization = true
        [build.internal]
        lists = [{centroids}]
        build_threads = {workers}
        spherical_centroids = {spherical_centroids} 
        """
        with self.conn.cursor() as cursor:
            cursor.execute(f"SET max_parallel_maintenance_workers TO {workers}")
            cursor.execute(f"SET max_parallel_workers TO {workers}")
            cursor.execute(
                f"CREATE INDEX {self.dataset}_rabitq ON {self.dataset}_corpus USING vchordrq (emb {ops}) WITH (options = $${ivf_config}$$)"
            )

            # Create a BM25 index
            cursor.execute(
                f"CREATE INDEX {self.dataset}_text_bm25 ON {self.dataset}_corpus USING bm25 (bm25 bm25_ops)"
            )

        logger.info("build index takes %f seconds", perf_counter() - start_time)

    def query(self, topk: int, distance: str):
        probe = int(0.1 * min(4 * int(self.num**0.5), self.num // 40))
        with self.conn.cursor() as cursor:
            cursor.execute(f"SET vchordrq.probes = {probe}")
        start_time = perf_counter()
        if distance == "euclidean":
            ops = "<->"
        elif distance == "cosine":
            ops = "<=>"
        else:
            ops = "<#>"
        with self.conn.cursor() as cursor:
            query = f"""
            WITH ranked_results AS (
                SELECT q.id AS qid, c.id, c.score
                FROM {self.dataset}_query q, 
                LATERAL (
                    SELECT id, {self.dataset}_corpus.emb {ops} q.emb AS score 
                    FROM {self.dataset}_corpus 
                    ORDER BY score
                    LIMIT {topk}
                ) c
            )
            SELECT qid, id, score
            FROM ranked_results
            ORDER BY qid, score DESC;
            """
            cursor.execute(
                query
            )
            res = cursor.fetchall()
        logger.info("query takes %f seconds", perf_counter() - start_time)
        return res
    
    def query_bm25(self, topk: int):
        start_time = perf_counter()
        with self.conn.cursor() as cursor:
            cursor.execute(
                f"SELECT q.id AS qid, c.id, c.score FROM {self.dataset}_query q, LATERAL ("
                f"SELECT id, {self.dataset}_corpus.bm25 <&> to_bm25query('{self.dataset}_text_bm25', q.text , '{self.dataset}_token') AS score "
                f"FROM {self.dataset}_corpus "
                f"ORDER BY score "
                f"LIMIT {topk}) c;"
            )
            res = cursor.fetchall()
        logger.info("query bm25 takes %f seconds", perf_counter() - start_time)
        return res

    def rrf(self, vector_results,bm25_results,topk: int, k: int = 60):
        start_time = perf_counter()
        rrf_scores = {}

        sorted_vector_results = {}
        for query_id, docs in vector_results.items():
            sorted_vector_results[str(query_id)] = dict(sorted(docs.items(), key=lambda x: x[1], reverse=True))

        sorted_bm25_results = {} 
        for query_id, docs in bm25_results.items():
            sorted_bm25_results[str(query_id)] = dict(sorted(docs.items(), key=lambda x: x[1], reverse=True))

        results = [sorted_vector_results, sorted_bm25_results]

        # Iterate through each ranking system
        for result in results:
            # Iterate through each document and its rank in current system
            for query_id, docs in result.items():
                for rank, (doc_id, _) in enumerate(docs.items(), start=1):
                    if query_id not in rrf_scores:
                        rrf_scores[query_id] = {}
                    if doc_id not in rrf_scores[query_id]:
                        rrf_scores[query_id][doc_id] = 0
                    # Calculate and accumulate RRF scores
                    rrf_scores[query_id][doc_id] += 1 / (k + rank)

        # Sort docs by RRF scores in descending order for each query
        sorted_results = {}
        for query_id, docs in rrf_scores.items():
            sorted_results[str(query_id)] = dict(sorted(docs.items(), key=lambda x: x[1], reverse=True)[:topk])
        
        logger.info("rrf rerank takes %f seconds", perf_counter() - start_time)
        return sorted_results

    def CorssEncoderReranker(self, results, documents, querys, model: str = "BAAI/bge-reranker-v2-m3"):
        start_time = perf_counter()
        bge_scores = {}
        def get_key_from_value(d, target_value):
            return next((v for k, v in d.items() if k == target_value), None)

        reranker = FlagReranker(
            'BAAI/bge-reranker-v2-m3', 
            query_max_length=256,
            passage_max_length=512,
            use_fp16=True,
            devices=["cuda:0"] # change ["cpu"] if you do not have gpu, but it will be very slow
        ) # Setting use_fp16 to True speeds up computation with a slight performance degradation

        for query_id, docs in tqdm(results.items()):
            pairs = []
            for doc_id in docs:
                if query_id not in bge_scores:
                    bge_scores[query_id] = {}
                val = get_key_from_value(documents, doc_id)
                pairs.append([get_key_from_value(querys, query_id), val["title"] + " " + val["text"]])
            scores = reranker.compute_score(pairs, normalize=True) 
            for i, doc_id in enumerate(docs):
                bge_scores[query_id][doc_id] = scores[i]

        # Sort docs by BGE scores in descending order for each query
        sorted_results = {}
        for query_id, docs in bge_scores.items():
            sorted_results[str(query_id)] = dict(sorted(docs.items(), key=lambda x: x[1], reverse=True))

        logger.info("bge rerank takes %f seconds", perf_counter() - start_time)
        return sorted_results


def main(dataset, topk, save_dir, dbname, user, password, host, port, vector_dim, limit, only_vector, only_bm25, distance):
    data_path = download_and_unzip(BASE_URL.format(dataset), save_dir)
    split = "dev" if dataset == "msmarco" else "test"
    corpus, query, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    num_doc = len(corpus)

    corpus_ids, corpus_text = [], []
    for i, (key, val) in enumerate(corpus.items()):
        if limit and i >= limit:
            break
        corpus_ids.append(key)
        corpus_text.append(val["title"] + " " + val["text"])

    qids, query_text = [], []
    for i, (key, val) in enumerate(query.items()):
        if limit and i >= limit:
            break
        qids.append(key)
        query_text.append(val)

    num_doc = len(corpus_ids) if limit else num_doc
    logger.info("Corpus: %d, query: %d", num_doc, len(qids))

    db_config = DBConfig(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port
    )

    client = PgClient(db_config, dataset, num_doc, vector_dim)
    # client.create()
    # client.insert(corpus_ids, corpus_text, qids, query_text)
    # client.index(int(len(os.sched_getaffinity(0)) * 0.8), distance)
    vector_results = client.query(topk, distance)
    bm25_results = client.query_bm25(topk)

    format_vector_results = {}
    for qid, cid, score in vector_results:
        key = str(qid)
        if key not in format_vector_results:
            format_vector_results[key] = {}
        # in vectorchord, cosine distance score = 1 - distance
        # need recovery
        format_vector_results[key][str(cid)] = 1-float(score)
    
    format_bm25_results = {}
    for qid, cid, score in bm25_results:
        key = str(qid)
        if key not in format_bm25_results:
            format_bm25_results[key] = {}
        # in vectorchord-bm25, score = - (real score)
        # need recovery
        format_bm25_results[key][str(cid)] = -float(score)

    merged_results = {}

    for qid, cid, _ in vector_results:
        key = str(qid)
        if key not in merged_results:
            merged_results[key] = set()
        merged_results[key].add(str(cid))

    for qid, cid, _ in bm25_results:
        key = str(qid)
        if key not in merged_results:
            merged_results[key] = set()
        merged_results[key].add(str(cid))

    def get_key_from_value(d, target_value):
        return next((v for k, v in d.items() if k == target_value), None)
    
    # cross-encoder reranker
    if not only_vector and not only_bm25:
        format_results = client.CorssEncoderReranker(merged_results, corpus, query, topk)

    # if not only_vector and not only_bm25:
    #     format_results = client.rrf(format_vector_results, format_bm25_results,topk, k=6)

    if only_vector:
        format_results = format_vector_results
    if only_bm25:
        format_results = format_bm25_results
    
    os.makedirs("results", exist_ok=True)
    with open(f"results/vectorchord_score_{dataset}.json", "w") as f:
        json.dump(format_results, f, indent=2)
    
    # format_text_results = {}
    
    # for qid, docs in format_results.items():
    #     format_text_results[get_key_from_value(query,qid)] = [(doc_id, get_key_from_value(corpus,doc_id)) for doc_id, _ in docs.items()] 
    # with open(f"results/vectorchord_text_{dataset}.json", "w") as f:
    #     json.dump(format_text_results, f, indent=2)
    del corpus, query

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        qrels, format_results, [1, 10, 100, 1000]
    )
    logger.info("NDCG: %s", ndcg)
    logger.info("Recall: %s", recall)
    logger.info("Precision: %s", precision)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    logger.info(args)
    main(**vars(args))
