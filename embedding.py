from __future__ import annotations

from FlagEmbedding import BGEM3FlagModel
class SentenceEmbedding:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=True,
        )
    def encode_docs(self, documents: list[str]):
        return self.model.encode(
            documents,
            batch_size=32,
            max_length=8192,
        )['dense_vecs']

    def encode_doc(self, doc: str):
        return self.model.encode(
            doc, 
            batch_size=32, 
            max_length=8192
        )['dense_vecs']

    def encode_queries(self, queries: list[str]):
        return self.model.encode(
            queries, 
            batch_size=32, 
            max_length=8192
        )['dense_vecs']

    def encode_query(self, query: str):
        return self.model.encode(
            query, 
            batch_size=32, 
            max_length=8192
        )['dense_vecs']


if __name__ == "__main__":
    se = SentenceEmbedding()
    text = "the quick brown fox jumps over the lazy dog"
    doc_emb = se.encode_doc(text)
    query_emb = se.encode_query(text)
    print(doc_emb.shape, query_emb.shape)
    docs_emb = se.encode_docs([text, text])
    queries_emb = se.encode_queries([text, text])
    print(docs_emb.shape, queries_emb.shape)
    print(doc_emb)
    # doc_emb.dump("doc_sentence_emb.np")
    # query_emb.dump("query_sentence_emb.np")
