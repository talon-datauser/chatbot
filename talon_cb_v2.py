# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Any
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns # Used for enhanced plotting
from google.cloud import bigquery
from google.oauth2.service_account import Credentials
from google.cloud.exceptions import GoogleCloudError

# Set Seaborn theme for better default aesthetics
sns.set_theme(style="whitegrid")

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="Talon Sight", layout="wide")
st.markdown("""
    <div style="background-color: #1F4E79; padding: 10px; margin-bottom: 10px;">
        <h1 style="color: white; text-align: center; margin: 0;">Talon Sight</h1>
    </div>
    """, unsafe_allow_html=True)

# --- Sidebar Actions ---
st.sidebar.image('https://cdn.prod.website-files.com/6748fc3886e2b1e2c003fbc9/6748fc3886e2b1e2c00420a6_Talon-Logistics-Color.png', width=150)

st.sidebar.header("Options")
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("\U0001F504 Refresh App"):
        st.cache_data.clear() # Clears all @st.cache_data caches
        st.session_state.clear() # Clears all session state variables
        st.rerun()
with col2:
    if st.button("\U0001F5D1️ Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.sql_history = []
        st.session_state.df_results = []
        st.session_state.chart_types = []
        st.rerun()

st.sidebar.markdown("### Example Questions")
st.sidebar.markdown("- Total revenue in 2023?")
st.sidebar.markdown("- Loads created each month?")
st.sidebar.markdown("- Average invoice amount?")
st.sidebar.markdown("- Top 5 customers by load count?")


# --- BigQuery Setup ---
try:
    creds_dict = json.loads(st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"])
    creds = Credentials.from_service_account_info(creds_dict)

    PROJECT = "talon-prod-2024"
    DATASET = "Temp_02"
    client = bigquery.Client(credentials=creds, project=PROJECT)
except Exception as e:
    st.error(f"BigQuery Initialization Error: Ensure GOOGLE_SERVICE_ACCOUNT_JSON is correctly set in .streamlit/secrets.toml. Details: {e}")
    st.stop()


# --- LangChain LLM ---
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )
except Exception as e:
    st.error(f"LLM Initialization Error: Ensure GOOGLE_API_KEY is correctly set in .streamlit/secrets.toml. Details: {e}")
    st.stop()

# --- Helper Functions for LLM Context ---

@st.cache_data(show_spinner="Describing BigQuery tables...")
def describe_table_columns(schema_df: pd.DataFrame, table_name: str, _llm_model: ChatGoogleGenerativeAI) -> str:
    """Generates human-friendly descriptions for table columns using an LLM."""
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
    response = _llm_model.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

@st.cache_data(show_spinner="Fetching table schemas...")
def get_table_schema_and_description(dataset: str, table: str, _llm_model: ChatGoogleGenerativeAI) -> Tuple[str, str]:
    """Fetches schema from BigQuery and generates descriptions."""
    query = f"""
        SELECT column_name, data_type
        FROM `{PROJECT}.{dataset}.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = '{table}'
    """
    df_schema = client.query(query).to_dataframe()
    schema_description = describe_table_columns(df_schema, table, _llm_model)
    schema_text = f"{table} Schema:\n" + "\n".join([
        f"- {row['column_name']} ({row['data_type']})" for _, row in df_schema.iterrows()
    ])

    return schema_text, schema_description

# Load schema and descriptions using cached function
schema_loads, loads_desc = get_table_schema_and_description(DATASET, "main_loads", llm)
schema_invoices, invoices_desc = get_table_schema_and_description(DATASET, "main_invoices", llm)


# --- LangGraph State ---
class ChatbotState(BaseModel):
    messages: List[BaseMessage]
    sql_guidelines: Optional[str] = None
    sql_query: Optional[str] = None
    sql_result: Optional[str] = None # CSV string of the result for LLM summary
    chart_type: Optional[str] = None
    df_current_result: Optional[pd.DataFrame] = None

    class Config:
        arbitrary_types_allowed = True


# --- Node: Generate Guidelines ---
def generate_guidelines(state: ChatbotState) -> ChatbotState:
    """Generates a step-by-step plan for SQL query generation."""
    user_question = state.messages[-1].content
    history = state.messages[-6:-1] if len(state.messages) > 6 else state.messages[:-1]
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
    Write 5–10 clear, concise bullet points that describe the logical steps.
    Each step should guide the SQL generation process.

    Consider the following aspects for your plan:
    - Which table(s) are relevant (`{PROJECT}.{DATASET}.main_loads`, `{PROJECT}.{DATASET}.main_invoices`)
    - Which columns will be queried and why (e.g., `load_id`, `invoice_amount`, `pickup_date`)
    - Any conditions or filters that should be applied (e.g., `WHERE year = 2023`, `WHERE status = 'completed'`)
    - How to handle date-related logic based on column types (e.g., using `PARSE_DATE`, `FORMAT_DATE`, `DATE_TRUNC`, `EXTRACT` functions specific to BigQuery)
    - What aggregations (e.g., `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`) or groupings (`GROUP BY`) are needed
    - Whether sorting (`ORDER BY`) or joins (`JOIN` across `main_loads` and `main_invoices` on common keys like `load_id`) are involved. If joining, specify the columns for the join.
    - How to limit results if the user asks for "top N" or "bottom N" (e.g., explicitly state "apply LIMIT N after ordering").
    - Any additional logic needed to get the correct result (e.g., distinct counts, subqueries if necessary).

    Be thoughtful and concise. Avoid SQL syntax — this is strictly the logical plan.

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
    """Generates a BigQuery Standard SQL query based on guidelines and user question."""
    user_question = state.messages[-1].content
    prompt = f"""
    You are a professional data analyst writing SQL queries for Google BigQuery using **Standard SQL**.

    Your job is to write a correct and efficient SQL query to answer the user's question using the schemas and guidelines below.

    Requirements:
    - Follow BigQuery Standard SQL syntax.
    - Ensure all functions, expressions, and date handling are supported in BigQuery (e.g., `PARSE_DATE`, `FORMAT_DATE`, `DATE_TRUNC`, `EXTRACT`, `CAST`).
    - Use fully qualified table names in the format `{PROJECT}.{DATASET}.table_name`.
    - **Crucially:** Always include an `ORDER BY` clause if the query involves aggregation over time or ranking (e.g., "top N", "each month", "highest", "lowest").
    - **Crucially:** Always include a `LIMIT` clause if the user asks for "top N" or "bottom N" results (e.g., `LIMIT 5`).
    - Output only a valid SQL query — no explanation, no commentary, no markdown text outside the code block.

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
    Return only a valid SQL query enclosed in triple backticks (` ```). Do not include any extra commentary.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    sql_match = re.search(r"```(?:sql)?(.*?)```", response.content, re.DOTALL)
    sql_code = sql_match.group(1).strip() if sql_match else response.content.strip()
    return ChatbotState(
        messages=state.messages,
        sql_guidelines=state.sql_guidelines,
        sql_query=sql_code
    )

# --- SQL Execution Helper Function ---
def _run_bigquery_query(sql: str) -> pd.DataFrame:
    """Internal helper to execute a BigQuery SQL query."""
    return client.query(sql).to_dataframe()

# --- Node: Interpret SQL Output ---
def interpret_sql(state: ChatbotState) -> ChatbotState:
    """
    Executes a generated SQL query, processes the result, summarizes the output,
    and determines the most suitable chart type.
    """
    try:
        with st.spinner("Executing query and fetching data..."):
            df_result = _run_bigquery_query(state.sql_query)

        if df_result.empty:
            return ChatbotState(messages=state.messages + [
                AIMessage(content="The SQL query executed successfully but returned no data. Please try refining your question or conditions.")
            ], sql_query=state.sql_query)

        df_cleaned = df_result.copy()
        str_cols_with_nan = df_cleaned.select_dtypes(include=['object', 'string']).columns[df_cleaned.select_dtypes(include=['object', 'string']).isnull().any()].tolist()
        if str_cols_with_nan:
            df_cleaned = df_cleaned.dropna(subset=str_cols_with_nan, how="all")

        # Attempt to convert relevant columns to appropriate types for plotting
        for col in df_cleaned.columns:
            # Try to convert object/string columns to numeric if all values are numeric
            if df_cleaned[col].dtype == 'object':
                try:
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                except ValueError:
                    pass # Keep as object if not convertible

            # Try to convert object/string columns to datetime if they look like dates
            if df_cleaned[col].dtype == 'object' or pd.api.types.is_datetime64_any_dtype(df_cleaned[col]):
                 # Only convert if at least one non-NaT value is present after coercion
                if pd.to_datetime(df_cleaned[col], errors='coerce').notna().any():
                    df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')


        match = re.search(r'(top|bottom)\s*(\d+)', state.messages[-1].content.lower())
        if match:
            top_n = int(match.group(2))
            df_cleaned = df_cleaned.head(top_n)

        state.sql_result = df_cleaned.to_csv(index=False)
        state.df_current_result = df_cleaned

        # --- Smarter Chart Type Suggestion Logic ---
        num_cols = df_cleaned.select_dtypes(include=["number", "int", "float"]).columns.tolist()
        cat_cols = df_cleaned.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        date_cols = df_cleaned.select_dtypes(include=["datetime", "M"]).columns.tolist()

        suggested_chart = 'table' # Default fallback

        # If data has a date column and a numerical column: Line or Area chart
        if len(date_cols) >= 1 and len(num_cols) >= 1:
            suggested_chart = 'line'
            # Could refine to 'area' if user asks for 'total over time' or 'cumulative'
            if any(word in state.messages[-1].content.lower() for word in ['total over time', 'cumulative', 'area']):
                suggested_chart = 'area'
        # If data has one categorical and one numerical column: Bar or Pie chart
        elif len(cat_cols) >= 1 and len(num_cols) >= 1:
            # Suggest Pie if user asks for proportion/share AND number of unique categories is small
            if any(word in state.messages[-1].content.lower() for word in ['proportion', 'share', 'percentage']) and \
               df_cleaned[cat_cols[0]].nunique() <= 10: # Limit categories for readability in pie chart
                suggested_chart = 'pie'
            else:
                suggested_chart = 'bar' # Default for categorical vs numerical
        # If data has two numerical columns: Scatter plot
        elif len(num_cols) >= 2:
            suggested_chart = 'scatter'
        # If data has only one numerical column: Histogram (or bar for value counts)
        elif len(num_cols) == 1 and len(cat_cols) == 0 and len(date_cols) == 0:
            suggested_chart = 'histogram' # To show distribution
        # If data has only one categorical column: Bar chart of counts
        elif len(cat_cols) >= 1 and len(num_cols) == 0 and len(date_cols) == 0:
            suggested_chart = 'bar' # To show counts of each category

        state.chart_type = suggested_chart

    except GoogleCloudError as e:
        error_message = f"❌ BigQuery Error ({e.code}): The query failed. This might be due to incorrect column names, table names, data type mismatches, or invalid BigQuery syntax in the generated SQL. Details: `{e.message}`"
        return ChatbotState(messages=state.messages + [
            AIMessage(content=error_message)
        ], sql_query=state.sql_query)

    except Exception as e:
        return ChatbotState(messages=state.messages + [
            AIMessage(content=f"❌ An unexpected error occurred during SQL execution or data processing: `{e}`")
        ], sql_query=state.sql_query)

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
    - Contain only plain text (no markdown, no bold, no italic, no special fonts).
    - Be well-written and professional.
    """
    with st.spinner("Summarizing results..."):
        summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
        # More robust regex to remove all common markdown characters
        summary_text = re.sub(r'[\*_`#]+', '', summary_response.content).strip()
        summary_text = re.sub(r'\s+', ' ', summary_text).strip() # Normalize whitespace
        summary_text = summary_text.replace('\\', '') # Remove backslashes if any escape characters are left


    return ChatbotState(
        messages=state.messages + [AIMessage(content=summary_text)],
        sql_query=state.sql_query,
        sql_result=state.sql_result,
        df_current_result=state.df_current_result,
        chart_type=state.chart_type
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
graph = builder.compile()


# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "sql_history" not in st.session_state:
    st.session_state.sql_history = []
if "df_results" not in st.session_state:
    st.session_state.df_results = []
if "chart_types" not in st.session_state:
    st.session_state.chart_types = []


# --- Chat Input and Execution ---
user_input = st.chat_input("Ask a question about your data...")
if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    current_state = ChatbotState(messages=st.session_state.chat_history)

    with st.spinner("Processing your request..."):
        try:
            result_state = graph.invoke(current_state)
            if isinstance(result_state, dict):
                result_state = ChatbotState(**result_state)

            st.session_state.chat_history = result_state.messages
            if result_state.sql_query:
                st.session_state.sql_history.append(result_state.sql_query)
            if result_state.df_current_result is not None:
                st.session_state.df_results.append(result_state.df_current_result)
            if result_state.chart_type:
                st.session_state.chart_types.append(result_state.chart_type)
            else:
                st.session_state.chart_types.append('table')

        except Exception as e:
            st.session_state.chat_history.append(AIMessage(content=f"❌ An internal processing error occurred: {e}"))
    st.rerun()


# --- Display Chat History, SQL, and Results ---
chat_container = st.container()

with chat_container:
    sql_index = 0
    chat = st.session_state.chat_history
    while sql_index * 2 < len(chat):
        user_msg = chat[sql_index * 2]
        with st.chat_message("user"):
            st.markdown(user_msg.content)

        ai_msg_idx = sql_index * 2 + 1
        if ai_msg_idx < len(chat):
            ai_msg = chat[ai_msg_idx]
            with st.chat_message("assistant"):
                st.markdown(ai_msg.content)

                if sql_index < len(st.session_state.sql_history):
                    with st.expander("Generated SQL Query"):
                        st.code(st.session_state.sql_history[sql_index], language="sql")

                    if sql_index < len(st.session_state.df_results):
                        df = st.session_state.df_results[sql_index]
                        st.subheader("📊 Query Result Table")
                        st.dataframe(df)

                        csv = df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Download Result as CSV",
                            data=csv,
                            file_name=f"query_result_{sql_index+1}.csv",
                            mime="text/csv",
                            key=f"download_csv_{sql_index}"
                        )

                        suggested_chart = st.session_state.chart_types[sql_index] if sql_index < len(st.session_state.chart_types) else 'table'

                        if not df.empty:
                            st.subheader(f"📈 Suggested Chart: {suggested_chart.replace('_', ' ').capitalize()}")

                            num_cols = df.select_dtypes(include=["number", "int", "float"]).columns.tolist()
                            cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
                            date_cols = df.select_dtypes(include=["datetime", "M"]).columns.tolist() # 'M' for datetime64

                            try:
                                if suggested_chart == 'bar':
                                    if len(cat_cols) >= 1 and len(num_cols) >= 1:
                                        x_col = cat_cols[0]
                                        y_col = num_cols[0]
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        sns.barplot(x=x_col, y=y_col, data=df, ax=ax, palette='viridis')
                                        ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
                                        ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=12)
                                        ax.set_title(f'{y_col.replace("_", " ").title()} by {x_col.replace("_", " ").title()}', fontsize=14)
                                        plt.xticks(rotation=45, ha='right', fontsize=10)
                                        plt.yticks(fontsize=10)
                                        ax.grid(axis='y', linestyle='--', alpha=0.7)
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                        plt.close(fig)
                                    elif len(num_cols) == 1: # Single numerical, show value counts
                                        x_col = num_cols[0]
                                        value_counts_df = df[x_col].value_counts().reset_index()
                                        value_counts_df.columns = [x_col, 'count']
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        sns.barplot(x=x_col, y='count', data=value_counts_df, ax=ax, palette='viridis')
                                        ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
                                        ax.set_ylabel("Count", fontsize=12)
                                        ax.set_title(f'Distribution of {x_col.replace("_", " ").title()}', fontsize=14)
                                        plt.xticks(rotation=45, ha='right', fontsize=10)
                                        plt.yticks(fontsize=10)
                                        ax.grid(axis='y', linestyle='--', alpha=0.7)
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                        plt.close(fig)
                                    elif len(cat_cols) >= 1: # Single categorical, show value counts
                                        x_col = cat_cols[0]
                                        value_counts_df = df[x_col].value_counts().reset_index()
                                        value_counts_df.columns = [x_col, 'count']
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        sns.barplot(x=x_col, y='count', data=value_counts_df, ax=ax, palette='viridis')
                                        ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
                                        ax.set_ylabel("Count", fontsize=12)
                                        ax.set_title(f'Counts by {x_col.replace("_", " ").title()}', fontsize=14)
                                        plt.xticks(rotation=45, ha='right', fontsize=10)
                                        plt.yticks(fontsize=10)
                                        ax.grid(axis='y', linestyle='--', alpha=0.7)
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                        plt.close(fig)
                                    else:
                                        st.info("Insufficient data for a meaningful bar chart.")

                                elif suggested_chart == 'line':
                                    if len(date_cols) >= 1 and len(num_cols) >= 1:
                                        x_col = date_cols[0]
                                        y_col = num_cols[0]
                                        fig, ax = plt.subplots(figsize=(12, 6))
                                        df_plot = df.dropna(subset=[x_col, y_col]).sort_values(by=x_col)
                                        sns.lineplot(x=x_col, y=y_col, data=df_plot, ax=ax, marker='o', color='royalblue')
                                        ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
                                        ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=12)
                                        ax.set_title(f'{y_col.replace("_", " ").title()} Over Time ({x_col.replace("_", " ").title()})', fontsize=14)
                                        plt.xticks(rotation=45, ha='right', fontsize=10)
                                        plt.yticks(fontsize=10)
                                        ax.grid(True, linestyle='--', alpha=0.7)
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                        plt.close(fig)
                                    else:
                                        st.info("Insufficient date and numerical data for a meaningful line chart.")

                                elif suggested_chart == 'scatter':
                                    if len(num_cols) >= 2:
                                        x_col = num_cols[0]
                                        y_col = num_cols[1]
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        sns.scatterplot(x=x_col, y=y_col, data=df, ax=ax, hue=cat_cols[0] if cat_cols else None, palette='deep')
                                        ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
                                        ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=12)
                                        ax.set_title(f'Relationship between {x_col.replace("_", " ").title()} and {y_col.replace("_", " ").title()}', fontsize=14)
                                        plt.xticks(fontsize=10)
                                        plt.yticks(fontsize=10)
                                        ax.grid(True, linestyle='--', alpha=0.7)
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                        plt.close(fig)
                                    else:
                                        st.info("Need at least two numerical columns for a meaningful scatter chart.")

                                elif suggested_chart == 'pie':
                                    if len(cat_cols) >= 1 and len(num_cols) >= 1:
                                        cat_col = cat_cols[0]
                                        val_col = num_cols[0]
                                        # Aggregate data for pie chart: sum of value column by category
                                        df_pie_data = df.groupby(cat_col)[val_col].sum().nlargest(5) # Top 5 + 'Others'
                                        if len(df[cat_col].unique()) > 5:
                                            others_sum = df.groupby(cat_col)[val_col].sum().nsmallest(len(df[cat_col].unique()) - 5).sum()
                                            if others_sum > 0:
                                                df_pie_data['Others'] = others_sum

                                        if not df_pie_data.empty:
                                            fig, ax = plt.subplots(figsize=(9, 9))
                                            wedges, texts, autotexts = ax.pie(df_pie_data, labels=df_pie_data.index,
                                                                              autopct='%1.1f%%', startangle=90,
                                                                              pctdistance=0.85, colors=sns.color_palette('pastel'))
                                            ax.axis('equal')
                                            ax.set_title(f'Proportion of {val_col.replace("_", " ").title()} by {cat_col.replace("_", " ").title()}', fontsize=14)
                                            plt.setp(autotexts, size=10, weight="bold")
                                            plt.setp(texts, size=10)
                                            plt.tight_layout()
                                            st.pyplot(fig)
                                            plt.close(fig)
                                        else:
                                            st.info("Not enough data for a meaningful pie chart.")
                                    else:
                                        st.info("Need one categorical and one numerical column for a meaningful pie chart.")

                                elif suggested_chart == 'area':
                                    if len(date_cols) >= 1 and len(num_cols) >= 1:
                                        x_col = date_cols[0]
                                        y_col = num_cols[0]
                                        fig, ax = plt.subplots(figsize=(12, 6))
                                        df_plot = df.dropna(subset=[x_col, y_col]).sort_values(by=x_col)
                                        sns.lineplot(x=x_col, y=y_col, data=df_plot, ax=ax, fill=True, color='lightseagreen')
                                        ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
                                        ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=12)
                                        ax.set_title(f'{y_col.replace("_", " ").title()} Area Chart Over Time ({x_col.replace("_", " ").title()})', fontsize=14)
                                        plt.xticks(rotation=45, ha='right', fontsize=10)
                                        plt.yticks(fontsize=10)
                                        ax.grid(True, linestyle='--', alpha=0.7)
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                        plt.close(fig)
                                    else:
                                        st.info("Insufficient date and numerical data for a meaningful area chart.")

                                elif suggested_chart == 'histogram':
                                    if len(num_cols) >= 1:
                                        x_col = num_cols[0]
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        sns.histplot(df[x_col].dropna(), kde=True, ax=ax, color='purple', bins=10) # 10 bins as a default
                                        ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
                                        ax.set_ylabel("Frequency", fontsize=12)
                                        ax.set_title(f'Distribution of {x_col.replace("_", " ").title()}', fontsize=14)
                                        plt.xticks(fontsize=10)
                                        plt.yticks(fontsize=10)
                                        ax.grid(axis='y', linestyle='--', alpha=0.7)
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                        plt.close(fig)
                                    else:
                                        st.info("No numerical columns for a histogram.")

                                else:
                                    st.info(f"Could not generate a specific chart for '{suggested_chart}'. Displaying table instead.")

                            except Exception as chart_e:
                                st.error(f"Error generating chart: {chart_e}. Displaying table instead.")
                        else:
                            st.info("Query returned an empty result, no chart to display.")
                    else:
                        st.info("No query result to display (likely due to an error during execution).")

        sql_index += 1

st.markdown("<script>window.scrollTo(0,document.body.scrollHeight);</script>", unsafe_allow_html=True)
