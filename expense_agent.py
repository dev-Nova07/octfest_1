# =============================================================
# expense_agent.py -- Expense Assistant FastAPI Application
# Week 16 -- Deploying & Monitoring Agentic AI Systems
# =============================================================

import os
import time
import json
import logging
from dotenv import load_dotenv

# Load .env before any other module reads environment variables
load_dotenv(override=True)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# =============================================================
# Structured JSON Logger
# JSON logs are parseable by Render, Datadog, CloudWatch, GCP Logging.
# =============================================================

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and key not in log_entry:
                log_entry[key] = value
        return json.dumps(log_entry)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger

logger = get_logger("expense_agent")

# =============================================================
# Storage
# A JSON file keeps this project simple. On Render's free tier the
# disk is ephemeral: expenses.json resets on every redeploy/restart.
# In production replace this with a real database.
# =============================================================

EXPENSE_FILE = os.getenv("EXPENSE_FILE", "expenses.json")

def load_expenses():
    if not os.path.exists(EXPENSE_FILE):
        return []
    with open(EXPENSE_FILE, "r") as file:
        return json.load(file)

def save_expenses(expenses):
    with open(EXPENSE_FILE, "w") as file:
        json.dump(expenses, file, indent=4)

if not os.path.exists(EXPENSE_FILE):
    save_expenses([])

# =============================================================
# Tools
# =============================================================

@tool
def add_expense(amount: float, category: str) -> str:
    """Add a new expense when the user wants to record money spent."""
    expenses = load_expenses()
    expenses.append({"amount": amount, "category": category.lower()})
    save_expenses(expenses)
    return f"Expense added successfully: Rs {amount} for {category}."


@tool
def get_expenses(category: str = "") -> str:
    """Get saved expenses or the total for a category."""
    expenses = load_expenses()
    if not expenses:
        return "No expenses have been recorded yet."
    if category:
        category = category.lower()
        expenses = [e for e in expenses if e["category"] == category]
    if not expenses:
        return f"No expenses found for category '{category}'."
    total = sum(e["amount"] for e in expenses)
    return f"Found {len(expenses)} expense(s). Total = Rs {total:.2f}"

# =============================================================
# Agent
# =============================================================

SYSTEM_PROMPT = (
    "You are a simple Expense Assistant.\n\n"
    "Your job is to help the user manage personal expenses.\n\n"
    "You have two tools:\n"
    "1. add_expense -- use when the user wants to record money they spent.\n"
    "2. get_expenses -- use when the user wants to know their expenses "
    "or how much they spent.\n\n"
    "Rules:\n"
    "- Use the appropriate tool for stored expense questions.\n"
    "- Do not invent expense information.\n"
    "- If the user gives an amount and category, record it.\n"
    "- Be concise and friendly.\n"
    "- If unrelated to expenses, say you can only help with expenses."
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)

memory = MemorySaver()

agent = create_react_agent(
    model=llm,
    tools=[add_expense, get_expenses],
    checkpointer=memory,
    prompt=SYSTEM_PROMPT,
)

logger.info("Expense agent initialised", extra={"model": "gpt-4o-mini"})

# =============================================================
# FastAPI Application
# =============================================================

app = FastAPI(
    title="Expense Assistant API",
    description="LangGraph-powered personal expense agent.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str = Field(
        ..., description="Unique conversation/session ID.", example="user1"
    )
    message: str = Field(
        ..., description="The user's expense message.",
        example="I spent 500 rupees on food"
    )

class ChatResponse(BaseModel):
    session_id: str
    response: str
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    service: str
    uptime_seconds: float
    model: str
    tools_available: list

SERVER_START_TIME = time.time()

@app.get("/health", response_model=HealthResponse, tags=["Operations"])
def health_check():
    """Health check used by Render and Cloud Run."""
    return HealthResponse(
        status="healthy",
        service="expense-agent",
        uptime_seconds=round(time.time() - SERVER_START_TIME, 2),
        model="gpt-4o-mini",
        tools_available=["add_expense", "get_expenses"],
    )

@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
def chat(request: ChatRequest):
    """Main chat endpoint. Reuse session_id across turns for memory."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    logger.info(
        "Incoming request",
        extra={"session_id": request.session_id, "message_length": len(request.message)},
    )
    start_time = time.time()
    try:
        config = {"configurable": {"thread_id": request.session_id}}
        result = agent.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config,
        )
        response_text = result["messages"][-1].content
        latency_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(
            "Request completed",
            extra={"session_id": request.session_id, "latency_ms": latency_ms},
        )
        return ChatResponse(
            session_id=request.session_id,
            response=response_text,
            latency_ms=latency_ms,
        )
    except Exception as e:
        logger.error(
            "Agent invocation failed",
            extra={"session_id": request.session_id, "error": str(e)},
        )
        raise HTTPException(
            status_code=500,
            detail="Expense agent is temporarily unavailable. Please try again.",
        )
