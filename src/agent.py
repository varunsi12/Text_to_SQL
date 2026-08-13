"""Agent logic for text-to-SQL conversion."""
import os
import re
import sqlite3
import json
import time
from openai import OpenAI
from src.utils import get_schema_context, load_db, query_db
from typing import List, Dict, Optional, Any, Tuple

DEFAULT_MODEL = "accounts/fireworks/models/deepseek-v4-flash"


class TextToSQLAgent:
    def __init__(self, model_id: Optional[str] = None, db_path: str = "data/Chinook.db"):
        self.api_key = os.getenv("FIREWORKS_API_KEY")
        if not self.api_key:
            raise ValueError("FIREWORKS_API_KEY environment variable is not set")

        self.model_id = model_id or os.getenv("FIREWORKS_MODEL") or DEFAULT_MODEL

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.fireworks.ai/inference/v1",
        )
        self.db_path = db_path
        self.conn = load_db(self.db_path)
        self.schema_context = get_schema_context(self.conn)
        
        # Initialize conversation state
        self.history: List[Dict[str, str]] = []

        # System prompt explicitly instructing JSON response for Fireworks API
        self.system_prompt = f"""
        You are an expert SQLite data analyst.
        Convert user natural-language questions into valid, executable SQLite queries.

        RULES:
        1. Respond ONLY with a valid JSON object containing a "sql" key:
        {{\"sql\": \"SELECT ...\"}}
        2. Generate READ-ONLY SQL queries (SELECT or WITH ... SELECT).
        3. Never write mutating operations (INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, REPLACE, ATTACH, DETACH, VACUUM).
        4. Use strictly the tables and columns present in the schema provided below.
        5. Prefer explicit JOIN conditions using foreign key relationships.

        DATABASE SCHEMA:
        {self.schema_context}
        """
    def reset_conversation(self) -> None:
        """Clears multi-turn conversation context for the /reset command."""
        self.history = []

    def validate_sql_safety(self, sql_query: str) -> bool:

        denied_keywords = ["UPDATE", "DELETE", "INSERT", "ALTER", "DROP", 
            "CREATE", "REPLACE", "ATTACH", "DETACH", "VACUUM"]
        sql_upper = sql_query.upper().strip()
        for keyword in denied_keywords:
            if re.search(rf"\b{keyword}\b", sql_upper):
                return False
        return True

    def clean_sql_output(self, text: str) -> str:
        """Fallback method to strip markdown code fences if JSON parsing fails."""
        text = text.strip()

        # 1. Strip Markdown backticks (e.g., ```sql or ```json)
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # 2. Extract "sql" key if structured as JSON
        try:
            data = json.loads(text)
            return data.get("sql", "").strip()
        except json.JSONDecodeError:
            return text  

    def generate_sql_query(self, question: str, error_context: Optional[str] = None) -> str:

        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.history[-8:],
        ]

        # If retrying after a failed SQL execution,
        # give the model the error so it can repair the query.
        if error_context:
            prompt = (
                "The previous SQL query failed. Correct it using the database schema.\n\n"
                f"Original question:\n{question}\n\n"
                f"Execution error:\n{error_context}"
            )
        else:
            prompt = question

        messages.append({
            "role": "user",
            "content": prompt,
        })

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or "{}"

        try:
            data = json.loads(content)
            sql = data.get("sql", "").strip()

        except json.JSONDecodeError:
            sql = self.clean_sql_output(content)

        return sql


    def query(self, question: str, max_retries: int = 2) -> Dict[str, Any]:
        """Executes question with attempt tracking, safety validation, and bounded repair loop."""
        start_time = time.perf_counter()
        error_context = None
        sql = ""
        rows = []
        error_msg = None
        retries_used = 0

        for attempt in range(max_retries + 1):
            retries_used = attempt  # 0 = initial run, 1 = first retry, 2 = second retry

            # 1. Generate SQL
            sql = self.generate_sql_query(question, error_context=error_context)

            # 2. Safety Validation
            if not self.validate_sql_safety(sql):
                error_msg = "Safety Validation Error: Query contains forbidden mutating statements."
                error_context = f"Attempted SQL:\n{sql}\nError:\n{error_msg}"
                continue

            # 3. SQLite Execution
            try:
                rows = query_db(self.conn, sql, return_as_df=False)
                error_msg = None
                break  # Execution succeeded
            except Exception as e:
                error_msg = str(e)
                error_context = f"Failed SQL:\n{sql}\nSQLite Error Details:\n{error_msg}"

        total_latency_s = time.perf_counter() - start_time

        # 4. Commit to history ONLY on success
        if error_msg is None:
            self.history.extend([
                {"role": "user", "content": question},
                {
                    "role": "assistant",
                    "content": json.dumps({"sql": sql})
                }
            ])

            self.history = self.history[-8:]

        return {
            "question": question,
            "sql": sql,
            "results": rows,
            "success": error_msg is None,
            "error": error_msg,
            "retries": retries_used,
            "latency_s": round(total_latency_s, 3),
        }