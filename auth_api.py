import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
from sqlalchemy import create_engine, text
FRONTEND_BUILD_DIR = "front_signup/build" 

app = FastAPI(title="AI Super Search Auth API", version="1.0.0")

app.mount(
    "/",
    StaticFiles(directory=FRONTEND_BUILD_DIR, html=True),
    name="static_frontend"
)

load_dotenv()

DATABASE_URL = (
    os.getenv("NEON_API_URL")
    or os.getenv("DATABASE_URL")
    or os.getenv("POSTGRES_URL")
)
if not DATABASE_URL:
    raise RuntimeError("Database URL not found. Set NEON_API_URL or DATABASE_URL in your environment.")


engine = create_engine(DATABASE_URL, pool_pre_ping=True)


app = FastAPI(title="AI Super Search Auth API", version="1.0.0")


# CORS for local dev frontends (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SignupPayload(BaseModel):
    institute: str
    studying: str
    username: str
    contact_number: str
    email: EmailStr


class SigninPayload(BaseModel):
    email: EmailStr


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/signup")
def signup(payload: SignupPayload):
    try:
        with engine.begin() as conn:
            result = conn.execute(
                text(
                    """
                    INSERT INTO aisupersearch_signup (institute, studying, username, contact_number, email)
                    VALUES (:institute, :studying, :username, :contact_number, :email)
                    ON CONFLICT (email) DO NOTHING
                    RETURNING user_id
                    """
                ),
                {
                    "institute": payload.institute,
                    "studying": payload.studying,
                    "username": payload.username,
                    "contact_number": payload.contact_number,
                    "email": payload.email,
                },
            )
            row = result.fetchone()
            created = bool(row)
            # If already exists, fetch id
            if not created:
                # Email exists -> return 409 Conflict
                raise HTTPException(status_code=409, detail="Email already exists. Please sign in.")
            else:
                user_id = row[0]
        return {"ok": True, "created": created, "user_id": user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/signin")
def signin(payload: SigninPayload):
    try:
        with engine.begin() as conn:
            row = conn.execute(
                text("SELECT user_id, username FROM aisupersearch_signup WHERE email = :email"),
                {"email": payload.email},
            ).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Email not found. Please sign up.")
            return {"ok": True, "user_id": row[0], "username": row[1]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


