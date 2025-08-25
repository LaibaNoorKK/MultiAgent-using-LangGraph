from flask import Flask, request, session,jsonify
from chat_history import get_chat_history
from main import run_supervisor  # your supervisor/SQL/internet agent handler
import os
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    user_id = session.get("user_id", "guest")

    if "session_id" not in session:
        session["session_id"] = os.urandom(8).hex()

    session_id = session["session_id"]

    # Load history from Postgres
    history, user_id, session_id = get_chat_history(session.get("user_id"), session.get("session_id"))

    # Add user message to history
    history.add_user_message(user_message)

    # Get AI reply
    assistant_reply = run_supervisor(user_message, history)

    # Save assistant reply
    history.add_ai_message(assistant_reply)

    return jsonify({"reply": assistant_reply})
