# helpers/topic_extractor.py

from langchain_openai import ChatOpenAI
from config import config

class TopicExtractor:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.2, model_name=config.OPENAI_MODEL_NAME)

    def extract_topics(self, document_content: str, n_topics: int = 3) -> list:
        prompt = f"""
        Based on the following document content, suggest {n_topics} specific topics that are directly mentioned or closely related to the content. These topics should be potential follow-up questions or areas of interest for someone reading about this subject.

        Document content:
        {document_content}

        Topics:
        1.
        2.
        3.
        """
        response = self.llm.invoke(prompt)
        topics = [topic.strip() for topic in response.content.split("\n") if topic.strip() and topic[0].isdigit()]
        return topics[:n_topics]

topic_extractor = TopicExtractor()
