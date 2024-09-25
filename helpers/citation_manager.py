# helpers/citation_manager.py
from typing import List, Dict

class CitationManager:
    @staticmethod
    def generate_citations(sources: List[Dict]) -> List[Dict]:
        citations = []
        for idx, source in enumerate(sources, start=1):
            citation = {
                "id": idx,
                "text": f"[{idx}] {source['metadata'].get('source', 'Unknown')}",
                "content": source.get('content', ''),
                "metadata": source['metadata']
            }
            citations.append(citation)
        return citations

citation_manager = CitationManager()
