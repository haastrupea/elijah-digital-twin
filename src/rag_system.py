from pathlib import Path

from annotated_types import doc
import chromadb

from src.injest import load_documents_from_directory


class RAGSystem:
    chunk_size = 500
    chunk_overlap = 50
    def __init__(self, root_path: str | Path, collection = "knowledge_base", chunk_size: int = None, chunk_overlap: int= None, refresh: bool = False) -> None:
    
        self.collection_name = collection
        self.root_dir=  root_path
        self.db_path=  Path(root_path).resolve() / 'data' / "vector_store"
   
        self.documents = []

        self.chromadb_client = chromadb.PersistentClient(path=self.db_path)

        if chunk_size:
            self.chunk_size = chunk_size

        if chunk_overlap:
            self.chunk_overlap = chunk_overlap
        
        # try to build index from db
        db_index = self.build_index_from_db() if not refresh else None

        if not db_index:
            #setup db from scratch
           raw_docs =  self.extract_file_content()
           self.setup_db_documents(raw_docs)

    def extract_file_content(self) -> None:
        raw_dir = self.root_dir / "data" / "raw"

        documents = None
        try:
            documents = load_documents_from_directory(raw_dir)
        except FileNotFoundError as e:
            print(e)

        if not documents:
            print("No documents loaded from data/raw (add non-empty .pdf or .txt files).")
        
        return documents

    def prepare_chunk(self, text: str) -> list[str]:
        size = self.chunk_size
        overlap = self.chunk_overlap

        print(f"Indexing documents with chunk size={size}, overlap={overlap}")

        bag_of_words = text.split()

        chunks = []
        total_word = len(bag_of_words)
        next_chunk_start = size - overlap
        for i in range(0, total_word, next_chunk_start):
            chunk_stop = i + size
            chunk = ' '.join(bag_of_words[i:chunk_stop])
            if chunk:
                chunks.append(chunk)
        return chunks

    def build_index_from_file_content(self, docs: dict[str, str]):
        all_chunks = []
        for doc_id, content in docs.items():
            chunks = self.prepare_chunk(content)
            for idx, chunk in enumerate(chunks):
                all_chunks.append({ "id": f"{doc_id}_{idx}",  "text": chunk,  "source": doc_id, "chunk_idx": idx })

        self.documents = all_chunks
        return all_chunks

    def build_index_from_db(self):
        collection = self.get_collection()
        results = collection.get()
        if not results:
            return None

        print("building rag index from chromadb content")
        ids = results["ids"]
        docs = results["documents"]
        all_chunks = []
        for doc_id, chunk in zip(ids, docs):
                all_chunks.append({ "id": f"{doc_id}",  "text": chunk,  "source": doc_id, "chunk_idx": doc_id })
        
        self.documents = all_chunks

        return all_chunks

    def setup_db_documents(self, docs: dict[str, str]):
        collection_name = self.collection_name
        
        all_chunks = self.build_index_from_file_content(docs)
        if not all_chunks:
            raise ValueError("No text chunks created from documents. Please check your document content.")
        
        try:
            self.chromadb_client.delete_collection(collection_name)
        except:
            pass
        
        self.collection = self.chromadb_client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
        
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            self.collection.add(
                documents=[doc["text"] for doc in batch],
                ids=[doc["id"] for doc in batch],
                metadatas=[{"source": doc["source"], "chunk_idx": doc["chunk_idx"]} for doc in batch]
            )
    
    def get_collection(self, collection: str = None):
        collection_name = self.collection_name if not collection else collection
        db_connection = self.chromadb_client
        return db_connection.get_collection(name=collection_name)

    def audit_rag_content(self):
        collection = self.get_collection()

        if not collection:
            return {}
        
        doc_count = collection.count()
        sample = collection.peek(2)
        return {"doc_count": doc_count, "sample": sample}

    def retrieve(self, query: str, top_k: int = 10):
        collection = self.chromadb_client.get_collection(name=self.collection_name)
        
        results = collection.query(query_texts=[query], n_results=top_k)

        retrieved = []
        for i, doc_id in enumerate(results["ids"][0]):
            doc = next((d for d in self.documents if d["id"] == doc_id), None)
            if doc:
                distance = results["distances"][0][i]
                similarity = 1 / (1 + distance)
                retrieved.append((doc, similarity))

        all_results = {}
        for doc, score in retrieved:
            doc_id = doc["id"]
            if doc_id not in all_results:
                all_results[doc_id] = (doc, 0.0)
                all_results[doc_id] = (doc, all_results[doc_id][1] + score)
        

        aggregated = list(all_results.values())
        aggregated.sort(key=lambda x: x[1], reverse=True)

        return [{"retrieval_score": score, **doc} for doc, score in aggregated]