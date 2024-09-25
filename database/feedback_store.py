# database/feedback_store.py
import sqlite3
from typing import List, Dict, Tuple
import threading
import logging

logger = logging.getLogger(__name__)

class FeedbackStore:
    def __init__(self, db_path: str = "feedback.db"):
        self.db_path = db_path
        self.local = threading.local()
        self.create_table()

    def get_connection(self):
        if not hasattr(self.local, 'conn'):
            self.local.conn = sqlite3.connect(self.db_path)
        return self.local.conn

    def create_table(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL,
            feedback INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS topic_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            relevance INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        conn.commit()

    def add_feedback(self, document_id: str, feedback: int):
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO document_feedback (document_id, feedback) VALUES (?, ?)",
                (document_id, feedback)
            )
            conn.commit()
            logger.info(f"Added feedback for document {document_id}: {feedback}")
        except Exception as e:
            logger.error(f"Error adding feedback: {str(e)}")
            conn.rollback()

    def add_topic_feedback(self, topic: str, relevance: int):
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO topic_feedback (topic, relevance) VALUES (?, ?)",
                (topic, relevance)
            )
            conn.commit()
            logger.info(f"Added feedback for topic {topic}: {relevance}")
        except Exception as e:
            logger.error(f"Error adding topic feedback: {str(e)}")
            conn.rollback()

    def get_document_feedback(self, document_id: str) -> List[Tuple[int, str]]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT feedback, timestamp FROM document_feedback WHERE document_id = ? ORDER BY timestamp DESC",
            (document_id,)
        )
        return cursor.fetchall()

    def get_topic_stats(self) -> List[Dict]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
        SELECT topic, 
               COUNT(*) as suggestion_count, 
               AVG(relevance) as avg_relevance
        FROM topic_feedback
        GROUP BY topic
        ORDER BY suggestion_count DESC, avg_relevance DESC
        ''')
        results = cursor.fetchall()
        return [
            {
                "topic": row[0],
                "suggestion_count": row[1],
                "avg_relevance": row[2]
            }
            for row in results
        ]

    def get_most_helpful_documents(self, limit: int = 10) -> List[Tuple[str, int]]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
        SELECT document_id, SUM(feedback) as total_feedback
        FROM document_feedback
        GROUP BY document_id
        ORDER BY total_feedback DESC
        LIMIT ?
        ''', (limit,))
        return cursor.fetchall()

    def clear_feedback_data(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM document_feedback")
            cursor.execute("DELETE FROM topic_feedback")
            conn.commit()
            logger.info("Cleared all feedback data")
            return True
        except Exception as e:
            logger.error(f"Error clearing feedback data: {str(e)}")
            conn.rollback()
            return False

    def close_connection(self):
        if hasattr(self.local, 'conn'):
            self.local.conn.close()
            del self.local.conn

feedback_store = FeedbackStore()
