# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langgraph.graph import StateGraph
from pydantic import BaseModel
from typing import List, Optional
import re
import json
from google.cloud import bigquery
from google.oauth2.service_account import Credentials

# --- Streamlit UI ---
st.set_page_config(page_title="Talon Chatbot", layout="wide")
st.title("Talon Data Chatbot")
st.markdown("Ask questions about your load and invoice data.")

# --- Sidebar ---
st.sidebar.header("Options")
if st.sidebar.button("🔄 Refresh"):
    st.cache_data.clear()
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.sql_history = []

st.sidebar.markdown("### Example Questions")
st.sidebar.markdown("- Total revenue in 2023?")
st.sidebar.markdown("- Loads created each month?")
st.sidebar.markdown("- Average invoice amount?")

# --- BigQuery Setup ---
creds_dict = json.loads(st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"])
creds = Credentials.from_service_account_info(creds_dict)
PROJECT = "talon-prod-2024"
DATASET = "Temp_02"
client = bigquery.Client(credentials=creds, project=PROJECT)

# --- Load schema ---
def get_table_schema(dataset: str, table: str) -> str:
    query = f"""
        SELECT column_name, data_type 
        FROM `{PROJECT}.{dataset}.INFORMATION_SCHEMA.COLUMNS` 
        WHERE table_name = '{table}'
    """
    df_schema = client.query(query).to_dataframe()
    return f"{table} Schema:\n" + "\n".join(
        [f"- {row['column_name']} ({row['data_type']})" for _, row in df_schema.iterrows()]
    )

schema_loads = get_table_schema(DATASET, "main_loads")
schema_invoices = get_table_schema(DATASET, "main_invoices")

# --- LangChain LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=st.secrets["GOOGLE_API_KEY"]
)

# --- LangGraph State ---
class ChatbotState(BaseModel):
    messages: List[BaseMessage]
    sql_guidelines: Optional[str] = None
    sql_query: Optional[str] = None
    sql_result: Optional[str] = None

# --- Node: Generate Guidelines ---
def generate_guidelines(state: ChatbotState) -> ChatbotState:
    user_question = state.messages[-1].content

    history = state.messages[-4:-1] if len(state.messages) > 3 else state.messages[:-1]
    memory_snippets = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            memory_snippets.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            memory_snippets.append(f"Assistant: {msg.content}")
    memory_context = "\n".join(memory_snippets) if memory_snippets else "None"

    prompt = f"""
You are a data analyst assistant. For the user question below, write clear guidelines that explain what the SQL query needs to do. Include:
- Which table(s) to use
- What filters to apply
- What aggregation or columns to retrieve
- Any date formatting or group by logic

Use the table schemas provided below. If a user term does not exactly match a column name, choose the closest semantically similar column.

===================
📊 Table Schemas
===================
{schema_loads}

{schema_invoices}

===================
🧠 Previous Conversation
===================
{memory_context}

===================
❓ User Question
===================
{user_question}

===================
📄 Output
===================
List the SQL query guidelines in bullet points.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return ChatbotState(messages=state.messages, sql_guidelines=response.content)

# --- Node: Generate SQL ---
def generate_sql(state: ChatbotState) -> ChatbotState:
    user_question = state.messages[-1].content
    prompt = f"""
You are a data analyst. Use the following SQL writing guidelines and table schemas to write a BigQuery SQL query that answers the user's question.

Important:
- Always use fully qualified table names in the format `project.dataset.table`, for example: `{PROJECT}.{DATASET}.main_loads` and `{PROJECT}.{DATASET}.main_invoices`.

- Do not refer to unqualified table names like just `loads` or `invoices`.

===================
✍️ SQL Writing Guidelines
===================
{state.sql_guidelines}

===================
📊 Table Schemas
===================
{schema_loads}

{schema_invoices}

===================
❓ User Question
===================
{user_question}

===================
✅ Your Output
===================
Return only a valid SQL query enclosed in triple backticks, without any explanation.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    sql_match = re.search(r"```(?:sql)?(.*?)```", response.content, re.DOTALL)
    sql_code = sql_match.group(1).strip() if sql_match else response.content.strip()
    return ChatbotState(messages=state.messages, sql_guidelines=state.sql_guidelines, sql_query=sql_code)

# --- SQL Execution ---
def run_query(sql: str) -> pd.DataFrame:
    return client.query(sql).to_dataframe()

# --- Node: Interpret SQL Output ---
def interpret_sql(state: ChatbotState) -> ChatbotState:
    try:
        df_result = run_query(state.sql_query)
        str_cols = df_result.select_dtypes(include=["object", "string"]).columns.tolist()
        df_cleaned = df_result.dropna(subset=str_cols, how="all") if str_cols else df_result.dropna()

        match = re.search(r'top\s*(\d+)', state.messages[-1].content.lower())
        if match:
            top_n = int(match.group(1))
            df_cleaned = df_cleaned.head(top_n)

        if df_cleaned.empty:
            return ChatbotState(messages=state.messages + [
                AIMessage(content="⚠️ The query returned only missing or incomplete data. No meaningful output to summarize.")
            ])

        state.sql_result = df_cleaned.to_csv(index=False)
    except Exception as e:
        return ChatbotState(messages=state.messages + [
            AIMessage(content=f"❌ Error running SQL query:\n\n{e}")
        ])

    prompt = f"""
You are a data assistant. A SQL query has been run and here is the result:

===================
📋 SQL Query
===================
{state.sql_query}

===================
📊 Query Result (CSV)
===================
{state.sql_result}

===================
💬 User Question
===================
{state.messages[-1].content}

===================
🧠 Your Task
===================
Provide a helpful natural language summary that answers the user's question based only on this result.
Ensure that the number of items (like top 5) matches the user's request, excluding any rows with missing values.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return ChatbotState(messages=state.messages + [AIMessage(content=response.content)],
                        sql_query=state.sql_query, sql_result=state.sql_result)

# --- LangGraph Setup ---
builder = StateGraph(ChatbotState)
builder.add_node("generate_guidelines", generate_guidelines)
builder.add_node("generate_sql", generate_sql)
builder.add_node("interpret_sql", interpret_sql)
builder.set_entry_point("generate_guidelines")
builder.add_edge("generate_guidelines", "generate_sql")
builder.add_edge("generate_sql", "interpret_sql")
builder.set_finish_point("interpret_sql")
graph = builder.compile().with_config()

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "sql_history" not in st.session_state:
    st.session_state.sql_history = []

# --- Chat Input ---
user_input = st.chat_input("Ask a question about your data")
if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    current_state = ChatbotState(messages=st.session_state.chat_history)

    try:
        result_state = graph.invoke(current_state)
        if isinstance(result_state, dict):
            result_state = ChatbotState(**result_state)

        st.session_state.chat_history = result_state.messages

        # Save the SQL for this step
        if result_state.sql_query:
            st.session_state.sql_history.append(result_state.sql_query)

    except Exception as e:
        st.session_state.chat_history.append(AIMessage(content=f"❌ Error during execution: {e}"))


# --- Display Question + Answer + SQL Query (clean format) ---
sql_index = 0
chat = st.session_state.chat_history
while sql_index < len(st.session_state.sql_history):
    user_msg = chat[sql_index * 2]
    ai_msg = chat[sql_index * 2 + 1]

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_msg.content)

    # Show assistant answer
    with st.chat_message("assistant"):
        st.markdown(ai_msg.content)

    # Show SQL in expander
    with st.expander("Generated SQL"):
        st.code(st.session_state.sql_history[sql_index], language="sql")

    sql_index += 1
