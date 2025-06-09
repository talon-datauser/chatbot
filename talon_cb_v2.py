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
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import bigquery
from google.oauth2.service_account import Credentials

# --- Streamlit UI ---
st.set_page_config(page_title="Talon Chatbot", layout="wide")
st.title("Talon Data Chatbot")
st.markdown("Ask questions about your load and invoice data.")

st.sidebar.image('https://image.pitchbook.com/bRWQlec8vZYlhSNUSJSZ0H64MEH1705906265236_200x200', width=200)
st.sidebar.header("Options")
if st.sidebar.button("\U0001F504 Refresh"):
    st.cache_data.clear()
if st.sidebar.button("\U0001F5D1️ Clear Chat"):
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

# --- LangChain LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=st.secrets["GOOGLE_API_KEY"]
)

def describe_table_columns(schema_df: pd.DataFrame, table_name: str, llm) -> str:
    cols = schema_df.to_dict(orient="records")
    prompt = f"""
You are a data documentation assistant helping analysts understand BigQuery tables.
Your task is to generate a detailed 2–3 line human-friendly description for each column in the table `{table_name}`.

For **each column**, your description must:
- Explain what the column likely represents in a business context.
- Mention how the column may be used in analytics or reporting.
- Include **keywords or synonyms** that people might use when referring to this column (e.g. "revenue", "date", "ID", "location", etc.).
- Make **educated guesses** if names are ambiguous or unclear based on column name and data type.

Use this format:
- `column_name` (data_type): description and common terms...

Example:
- `invoice_amount` (FLOAT): Represents the total amount charged on an invoice. Useful for calculating revenue, profit, or cost. Commonly referred to as "revenue", "amount", or "charges".

Here are the columns for `{table_name}`:
{json.dumps(cols, indent=2)}

Now respond only with the descriptions in the format above, one line per column.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


# --- Load schema and descriptions ---
def get_table_schema_and_description(dataset: str, table: str, llm) -> tuple[str, str]:
    query = f"""
        SELECT column_name, data_type 
        FROM `{PROJECT}.{dataset}.INFORMATION_SCHEMA.COLUMNS` 
        WHERE table_name = '{table}'
    """
    df_schema = client.query(query).to_dataframe()
    schema_text = f"{table} Schema:\n" + "\n".join([
        f"- {row['column_name']} ({row['data_type']})" for _, row in df_schema.iterrows()
    ])
    schema_description = describe_table_columns(df_schema, table, llm)
    return schema_text, schema_description

schema_loads, loads_desc = get_table_schema_and_description(DATASET, "main_loads", llm)
schema_invoices, invoices_desc = get_table_schema_and_description(DATASET, "main_invoices", llm)

# --- LangGraph State ---
class ChatbotState(BaseModel):
    messages: List[BaseMessage]
    sql_guidelines: Optional[str] = None
    sql_query: Optional[str] = None
    sql_result: Optional[str] = None
    chart_type: Optional[str] = None 

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
You are a senior data analyst working on Google BigQuery using Standard SQL.

Your job is to interpret the user’s data question and break it down into a step-by-step plan to write the correct SQL query.

This plan should align with:
- The structure and data types of the available tables
- Standard SQL conventions as used in BigQuery
- Business context inferred from the table and column descriptions

===================
🧠 Your Goal
===================
Write 5–10 bullet points that describe:
- Which table(s) are relevant
- Which columns will be queried and why
- Any conditions or filters that should be applied
- How to handle date-related logic based on the column types
- What aggregations (e.g., COUNT, SUM, AVG) or groupings are needed
- Whether sorting or joins are involved
- Any additional logic needed to get the correct result

Be thoughtful and concise. Avoid SQL syntax — this is just the logic plan.

===================
📊 Available Tables & Columns
===================
{schema_loads}

{loads_desc}

{schema_invoices}

{invoices_desc}

===================
🧠 Previous Conversation (optional)
===================
{memory_context}

===================
❓ User Question
===================
{user_question}

===================
✅ Output Format
===================
- Bullet point 1
- Bullet point 2
...
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return ChatbotState(messages=state.messages, sql_guidelines=response.content)


# --- Node: Generate SQL ---
def generate_sql(state: ChatbotState) -> ChatbotState:
    user_question = state.messages[-1].content
    prompt = f"""
You are a professional data analyst writing SQL queries for Google BigQuery using **Standard SQL**.

Your job is to write a correct and efficient SQL query to answer the user's question using the schemas and guidelines below.

Requirements:
- Follow BigQuery Standard SQL syntax.
- Ensure all functions, expressions, and date handling are supported in BigQuery.
- Use fully qualified table names in the format `{PROJECT}.{DATASET}.table_name`.
- Output only a valid SQL query — no explanation.

===================
📋 SQL Guidelines
===================
{state.sql_guidelines}

===================
📊 Table Schemas + Descriptions
===================
{schema_loads}

{loads_desc}

{schema_invoices}

{invoices_desc}

===================
❓ User Question
===================
{user_question}

===================
✅ Output
===================
Return only a valid SQL query enclosed in triple backticks (` ``` `). Do not include any extra commentary.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    sql_match = re.search(r"```(?:sql)?(.*?)```", response.content, re.DOTALL)
    sql_code = sql_match.group(1).strip() if sql_match else response.content.strip()
    return ChatbotState(
        messages=state.messages,
        sql_guidelines=state.sql_guidelines,
        sql_query=sql_code
    )

# --- SQL Execution ---
def run_query(sql: str) -> pd.DataFrame:
    return client.query(sql).to_dataframe()

# --- Node: Interpret SQL Output ---
def interpret_sql(state: ChatbotState) -> ChatbotState:
    """
    Executes a generated SQL query, processes the result, summarizes the output, and determines the most suitable chart type.
    Ensures:
    - Clean and plain-text response (no formatting or markdown)
    - Semantic chart type suggestion based on both user query and result
    - Handles missing or incomplete data safely
    """
    try:
        # Run the query generated by the LLM
        df_result = run_query(state.sql_query)

        # Drop rows where all string columns are NaN to clean up the result
        str_cols = df_result.select_dtypes(include=["object", "string"]).columns.tolist()
        df_cleaned = df_result.dropna(subset=str_cols, how="all") if str_cols else df_result.dropna()

        # If the user asked for top N rows (e.g., "top 5"), truncate the result accordingly
        match = re.search(r'top\s*(\d+)', state.messages[-1].content.lower())
        if match:
            top_n = int(match.group(1))
            df_cleaned = df_cleaned.head(top_n)

        # If no useful data returned
        if df_cleaned.empty:
            return ChatbotState(messages=state.messages + [
                AIMessage(content="The SQL query returned no meaningful data to summarize.")
            ])

        # Convert the cleaned result to CSV
        state.sql_result = df_cleaned.to_csv(index=False)
        st.session_state.df_results.append(df_cleaned)

    except Exception as e:
        return ChatbotState(messages=state.messages + [
            AIMessage(content=f"Error while executing the SQL query:\n{e}")
        ])

    # Step 1: Generate a clean summary without any styling or markdown formatting
    summary_prompt = f"""
You are a helpful data assistant.
A SQL query has been run and the result is shown below.

User Question:
{state.messages[-1].content}

Query Result (CSV):
{state.sql_result}

Task:
Summarize the result in plain, clear, natural language.
The answer must:
- Directly answer the user's question based on the result
- Contain only plain text (no markdown, no bold, no italic, no special fonts)
- Be well-written and professional
"""

    summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
    summary_text = re.sub(r"[\\*_`]+", "", summary_response.content.strip())  # Remove any accidental styling

    # Step 2: Suggest appropriate chart type based on both user question and data shape
    preview_csv = "\n".join(state.sql_result.strip().splitlines()[:12])

    chart_prompt = f"""
You are a data visualization expert.

Analyze the following:
- The user's question
- The shape and structure of the SQL result (first 10 rows below)

User Question:
{state.messages[-1].content}

SQL Result Preview:
{preview_csv}

Based on this, what is the most suitable chart type to visualize the answer?
Respond only with a single chart type (e.g., bar, line, pie, scatter, heatmap, histogram, table).
"""

    chart_response = llm.invoke([HumanMessage(content=chart_prompt)])
    suggested_chart = chart_response.content.strip().lower()

    # Store the chart suggestion in session state for use in visualization
    if "chart_types" not in st.session_state:
        st.session_state.chart_types = []
    st.session_state.chart_types.append(suggested_chart)

    # Return updated chatbot state with the summary and SQL result
    return ChatbotState(
        messages=state.messages + [AIMessage(content=summary_text)],
        sql_query=state.sql_query,
        sql_result=state.sql_result
    )


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
if "df_results" not in st.session_state:
    st.session_state.df_results = []

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
        if result_state.sql_query:
            st.session_state.sql_history.append(result_state.sql_query)
    except Exception as e:
        st.session_state.chat_history.append(AIMessage(content=f"❌ Error during execution: {e}"))

# --- Display Chat + SQL ---
sql_index = 0
chat = st.session_state.chat_history
while sql_index < len(st.session_state.sql_history):
    user_msg = chat[sql_index * 2]
    ai_msg = chat[sql_index * 2 + 1]
    with st.chat_message("user"):
        st.markdown(user_msg.content)
    with st.chat_message("assistant"):
        st.markdown(ai_msg.content)
    with st.expander("Generated SQL"):
        st.code(st.session_state.sql_history[sql_index], language="sql")
    # Show table result (optional preview)
    if len(st.session_state.df_results) > sql_index:
        df = st.session_state.df_results[sql_index]
        st.dataframe(df)

        # --- Download CSV ---
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Result as CSV",
            data=csv,
            file_name=f"query_result_{sql_index+1}.csv",
            mime="text/csv",
        )

        # --- Basic Visualization ---
        if not df.empty and df.select_dtypes(include="number").shape[1] >= 1:
            st.subheader("📊 Result Chart")
            plt.figure(figsize=(10, 5))
            num_cols = df.select_dtypes(include="number").columns.tolist()
            sns.barplot(data=df[num_cols].head(10))
            plt.xticks(rotation=45)
            st.pyplot(plt.gcf())

    sql_index += 1
