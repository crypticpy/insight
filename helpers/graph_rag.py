import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


class DocumentGraph:
    def __init__(self, vectorstore_manager):
        self.graph = nx.Graph()
        self.vectorstore_manager = vectorstore_manager
        self.build_graph()

    def build_graph(self):
        documents = self.vectorstore_manager.get_all_documents()
        embeddings = [self.vectorstore_manager._embeddings.embed_query(doc.page_content) for doc in documents]

        for i, doc1 in enumerate(documents):
            self.graph.add_node(doc1.metadata['doc_id'], document=doc1)
            for j, doc2 in enumerate(documents[i + 1:], start=i + 1):
                similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                if similarity > 0.5:  # Adjust this threshold as needed
                    self.graph.add_edge(doc1.metadata['doc_id'], doc2.metadata['doc_id'], weight=similarity)

    def get_related_documents(self, doc_id, depth=2):
        return nx.ego_graph(self.graph, doc_id, radius=depth)