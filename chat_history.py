# chat_history.py
import uuid
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from langchain_postgres import PostgresChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_SSLMODE = os.getenv("POSTGRES_SSLMODE", "require")
TABLE_NAME = os.getenv("TABLE_NAME", "chat_history")


def get_psycopg_connection():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        sslmode=POSTGRES_SSLMODE
    )


class SimplePostgresChatMessageHistory:
    """
    Minimal replacement for PostgresChatMessageHistory that avoids psycopg2.sql.Composed
    """
    def __init__(self, user_id, session_id, connection):
        self._user_id = str(user_id)
        self._session_id = str(session_id)
        self._connection = connection

    @property
    def messages(self):
        msgs = []
        query = f"SELECT role, content FROM {TABLE_NAME} WHERE session_id = %s ORDER BY created_at ASC"
        with self._connection.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (self._session_id,))
            rows = cur.fetchall()
            for row in rows:
                if row['role'] == 'user':
                    msgs.append(HumanMessage(content=row['content']))
                else:
                    msgs.append(AIMessage(content=row['content']))
        return msgs

    def add_user_message(self, message):
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message):
        self.add_message(AIMessage(content=message))

    def add_message(self, message):
        query = f"""
            INSERT INTO {TABLE_NAME} (user_id, session_id, role, content, created_at)
            VALUES (%s, %s, %s, %s, NOW())
        """
        role = 'user' if isinstance(message, HumanMessage) else 'assistant'
        with self._connection.cursor() as cur:
            cur.execute(query, (self._user_id, self._session_id, role, message.content))
        self._connection.commit()



def get_chat_history(user_id=None, session_id=None):
    if not user_id:
        user_id = f"guest-{uuid.uuid4()}"
    if not session_id:
        session_id = str(uuid.uuid4())

    conn = get_psycopg_connection()
    history = SimplePostgresChatMessageHistory(user_id, session_id, conn)
    return history, user_id, session_id


def get_user_chat_sessions(user_id):
    """
    Returns a list of chat sessions for a given user.
    Each session contains session_id, first_time, and title (first message content).
    """
    sessions = []
    try:
        with get_psycopg_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = f"""
                    SELECT session_id,
                           MIN(created_at) AS first_time,
                           (
                               SELECT content
                               FROM {TABLE_NAME} ch2
                               WHERE ch2.session_id = ch1.session_id
                                 AND ch2.role = 'user'
                               ORDER BY created_at ASC
                               LIMIT 1
                           ) AS title
                    FROM {TABLE_NAME} ch1
                    WHERE user_id = %s
                    GROUP BY session_id
                    ORDER BY first_time DESC;
                """
                cur.execute(query, (user_id,))
                sessions = cur.fetchall()
    except Exception as e:
        print(f"Error fetching chat sessions: {e}")
    return sessions
