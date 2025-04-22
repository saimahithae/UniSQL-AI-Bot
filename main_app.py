import streamlit as st
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import sqlite3
import traceback
from openai import OpenAI
from datetime import datetime

# Set page configuration
st.set_page_config(page_title="UniSQLBot", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

# Set dark theme globally
st.markdown("""
<style>
    .reportview-container {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .sidebar .sidebar-content {
        background-color: #2D2D2D;
        color: #FFFFFF;
    }
    .Widget>label {
        color: #FFFFFF;
    }
    .st-bb {
        background-color: #2D2D2D;
        color: #FFFFFF;
    }
    .st-at {
        background-color: #3D3D3D;
    }
    .st-cv {
        background-color: #1E1E1E;
    }
    /* Make the sidebar wider */
    [data-testid="stSidebar"] {
        min-width: 450px !important;
        max-width: 450px !important;
    }
    /* Hide the sidebar collapse button */
    .css-fblp2m {
        display: none !important;
    }
    /* Adjust the main content area */
    section[data-testid="stSidebarContent"] {
        background-color: #1E1E1E !important;
    }
    /* Styling the button */
    .stButton > button {
        width: 100%;
    }
    /* API key input field */
    .api-key-input {
        background-color: #3D3D3D !important;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'rerun_query' not in st.session_state:
    st.session_state['rerun_query'] = None
if 'current_db' not in st.session_state:
    st.session_state['current_db'] = None
if 'previous_upload' not in st.session_state:
    st.session_state['previous_upload'] = None
if 'db_just_uploaded' not in st.session_state:
    st.session_state['db_just_uploaded'] = False

# Function to load available databases
@st.cache_data
def load_available_databases():
    # This would be replaced with the actual path to your databases
    db_dir = "./databases"
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
        # For testing/demo purposes - create a sample database if none exists
        create_sample_database(os.path.join(db_dir, "sample_academic.sqlite"))
        
    # Get all SQLite database files - handle different extensions
    db_files = [f for f in os.listdir(db_dir) if f.endswith(('.sqlite', '.db', '.sqlite3', '.db3'))]
    
    # If no database files found, provide clear message
    if not db_files:
        st.sidebar.error("No SQLite databases found in the ./databases directory. Please add .sqlite, .db, .sqlite3, or .db3 files.")
    
    return db_files, db_dir

# Create a sample database for demonstration if no databases are found
def create_sample_database(db_path):
    """Create a sample academic database with some tables and data"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create author table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS author (
            aid INTEGER PRIMARY KEY,
            name TEXT,
            homepage TEXT,
            oid INTEGER
        )
        ''')
        
        # Create publication table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS publication (
            pid INTEGER PRIMARY KEY,
            title TEXT,
            year INTEGER,
            cid INTEGER,
            jid INTEGER
        )
        ''')
        
        # Create writes table (relationship between authors and publications)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS writes (
            aid INTEGER,
            pid INTEGER,
            PRIMARY KEY (aid, pid),
            FOREIGN KEY (aid) REFERENCES author(aid),
            FOREIGN KEY (pid) REFERENCES publication(pid)
        )
        ''')
        
        # Insert some sample data
        authors = [
            (1, "Jane Smith", "http://smith.edu", 101),
            (2, "John Doe", "http://doe.com", 102),
            (3, "Maria Garcia", "http://garcia.org", 103)
        ]
        
        publications = [
            (1, "An Introduction to Database Systems", 2022, 1, 1),
            (2, "Advanced SQL Techniques", 2023, 2, 2),
            (3, "Natural Language Processing for Databases", 2021, 1, 3)
        ]
        
        writes = [
            (1, 1),
            (2, 2),
            (1, 3),
            (3, 3)
        ]
        
        cursor.executemany("INSERT INTO author VALUES (?, ?, ?, ?)", authors)
        cursor.executemany("INSERT INTO publication VALUES (?, ?, ?, ?, ?)", publications)
        cursor.executemany("INSERT INTO writes VALUES (?, ?)", writes)
        
        conn.commit()
        conn.close()
        
        return True
    except Exception as e:
        print(f"Error creating sample database: {str(e)}")
        return False

# Function to get database schema
def get_db_schema(db_path):
    """Extract schema information from a SQLite database file"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if this is a valid SQLite database by trying a simple query
        try:
            cursor.execute("SELECT sqlite_version();")
        except sqlite3.DatabaseError as e:
            if "file is not a database" in str(e):
                raise Exception("The file does not appear to be a valid SQLite database.")
            else:
                raise e
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        schema = {}
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            schema[table_name] = [(col[1], col[2]) for col in columns]  # (column_name, data_type)
            
            # Get sample data (first 3 rows)
            try:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
                sample_data = cursor.fetchall()
                schema[f"{table_name}_sample"] = sample_data
            except sqlite3.OperationalError:
                # Handle case where table exists but can't be queried
                schema[f"{table_name}_sample"] = []
        
        conn.close()
        return schema
    except Exception as e:
        # If there's an error, try to close the connection
        try:
            if conn:
                conn.close()
        except:
            pass
        raise e

def get_openai_client():
    api_key = st.session_state.get('api_key', '')
    if api_key:
        return OpenAI(api_key=api_key)
    return None

def extract_sql_query(text):
    # Extract SQL query from OpenAI response
    pattern = r"```sql(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    pattern = r"```(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # If no code blocks found, try to find SQL statements directly
    sql_keywords = r"(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|WITH|FROM)"
    pattern = rf"{sql_keywords}.*?;+"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0).strip()
    
    return text

def process_natural_language_query(query, db_path, schema):
    client = get_openai_client()
    if not client:
        return "Please enter your OpenAI API key in the sidebar."
    
    # Create a prompt with database schema information
    schema_info = ""
    for table_name, columns in schema.items():
        if not table_name.endswith('_sample'):
            schema_info += f"Table: {table_name}\n"
            schema_info += "Columns:\n"
            for col_name, col_type in columns:
                schema_info += f"  - {col_name} ({col_type})\n"
            
            # Add sample data if available
            if f"{table_name}_sample" in schema:
                schema_info += "Sample data:\n"
                for row in schema[f"{table_name}_sample"][:3]:
                    schema_info += f"  {row}\n"
            schema_info += "\n"
    
    prompt = f"""You are UniSQLBot, an AI assistant that converts natural language queries into SQL queries.

    Database Schema:
    {schema_info}

    User Query: "{query}"
    
    Please generate a SQL query that answers this question. Follow these guidelines:
    1. Output ONLY the SQL query itself wrapped in ```sql``` tags
    2. Use only the tables and columns that exist in the database schema
    3. Always use proper SQL syntax with correct table and column names
    4. Use appropriate JOINs when querying data from multiple tables
    5. Add helpful comments in your SQL to explain complex parts
    6. If the query might return no data, structure it to still execute without errors
    7. Use proper SQL operators (=, >, <, LIKE, IN, etc.) as appropriate for the question
    8. If the query is about finding patterns in text, use SQL's pattern matching with LIKE or wildcards
    9. For numerical data, use appropriate aggregation functions as needed (SUM, AVG, COUNT, etc.)
    10. If the query contains ambiguities, make reasonable assumptions and note them in comments
    
    Respond with a valid SQL query only, wrapped in ```sql``` tags.
    """

    messages = [
        {"role": "system", "content": "You are UniSQLBot, an assistant for generating SQL queries from natural language."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def execute_sql_query(sql_query, db_path):
    """Execute a SQL query against the SQLite database and return the results"""
    result = {'success': False}
    conn = None
    
    try:
        # Validate database path
        if not os.path.exists(db_path):
            raise Exception(f"Database file not found at: {db_path}")
        
        # Connect to database with timeout to handle locked database issues
        conn = sqlite3.connect(db_path, timeout=10)
        
        # Execute the query
        df = pd.read_sql_query(sql_query, conn)
        
        # Determine if visualization is needed
        should_visualize = len(df) > 0 and len(df) <= 100 and df.select_dtypes(include=['number']).shape[1] > 0
        
        fig_data = None
        if should_visualize:
            # Set dark theme for plots
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#2D2D2D')
            ax.set_facecolor('#2D2D2D')
            
            # Basic visualization logic
            if len(df.columns) == 2 and df.select_dtypes(include=['number']).shape[1] == 1:
                # Single numeric column with a category -> bar chart
                numeric_col = df.select_dtypes(include=['number']).columns[0]
                category_col = [col for col in df.columns if col != numeric_col][0]
                df = df.sort_values(numeric_col)
                bars = sns.barplot(x=numeric_col, y=category_col, data=df, ax=ax, palette='viridis')
                # Ensure there's contrast for the labels
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                ax.tick_params(colors='white')
                ax.set_title(f"{category_col} vs {numeric_col}")
            
            elif len(df.columns) >= 2 and df.select_dtypes(include=['number']).shape[1] >= 2:
                # Two or more numeric columns -> scatter plot or line plot
                numeric_cols = df.select_dtypes(include=['number']).columns[:2]
                scatter = sns.scatterplot(x=numeric_cols[0], y=numeric_cols[1], data=df, ax=ax, palette='viridis')
                # Ensure there's contrast for the labels
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                ax.tick_params(colors='white')
                ax.set_title(f"{numeric_cols[1]} vs {numeric_cols[0]}")
            
            else:
                # Fallback to a simple bar chart of the first numeric column
                numeric_col = df.select_dtypes(include=['number']).columns[0]
                df[numeric_col].plot(kind='bar', ax=ax, color='#9D8CFF')
                # Ensure there's contrast for the labels
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                ax.tick_params(colors='white')
                ax.set_title(f"{numeric_col} Values")
            
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', facecolor=fig.get_facecolor())
            buf.seek(0)
            fig_data = buf.getvalue()
            plt.close(fig)
        
        # Convert DataFrame to string representation for display
        if len(df) > 0:
            result = {
                'success': True,
                'data': df,
                'output': df.to_string(),
                'figure': fig_data,
                'row_count': len(df)
            }
        else:
            result = {
                'success': True,
                'data': pd.DataFrame(),
                'output': "Query executed successfully, but returned no data.",
                'figure': None,
                'row_count': 0
            }
    
    except Exception as e:
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        result = {
            'success': False,
            'error': error_msg,
            'stack_trace': stack_trace
        }
    
    finally:
        # Always close the connection
        if conn:
            conn.close()
    
    return result

def check_is_chat_query(query):
    # Simple function to check if the query is just a greeting or general chat
    greetings = ["hi", "hello", "hey", "greetings", "howdy", "good morning", "good afternoon", "good evening"]
    chat_queries = ["how are you", "what's up", "what can you do", "help me", "tell me about"]
    
    query_lower = query.lower()
    for greeting in greetings:
        if query_lower.startswith(greeting):
            return True
    
    for chat_query in chat_queries:
        if chat_query in query_lower:
            return True
            
    return False

def handle_chat_query(query):
    # Handle general conversational queries
    if "what can you do" in query.lower() or "help" in query.lower():
        return "I'm UniSQLBot! I can help you convert natural language questions into SQL queries for your database. Just select a database from the sidebar and ask me questions about the data in plain English."
    
    if "about" in query.lower() and "you" in query.lower():
        return "I'm UniSQLBot, a natural language to SQL conversion tool developed by Sai Mahitha, Ajay, and Sai Rathnakar. I can help you query databases without knowing SQL."
    
    # Default greeting response
    return "Hello! I'm UniSQLBot. Please select a database from the sidebar and ask me questions about the data in plain English."

def main():
    # Sidebar for settings
    with st.sidebar:
        # Clear cache button
        if st.button("Clear Cache and Restart"):
            st.cache_data.clear()

        # Settings header
        st.markdown("## Settings")
        
        # API key input
        api_key = st.text_input("Enter your OpenAI API key:", type="password", key="api_key_input")
        if api_key:
            st.session_state['api_key'] = api_key
            st.success("API key saved!")
        
        # Database selection
        st.markdown("---")
        st.markdown("## Select Database")
        db_files, db_dir = load_available_databases()
        
        if db_files:
            selected_db = st.selectbox("Select Database:", db_files)
            if selected_db:
                db_path = os.path.join(db_dir, selected_db)
                st.session_state['current_db'] = db_path
                
                # Check if database is accessible
                try:
                    schema = get_db_schema(db_path)
                    
                    # Display database info
                    tables = [table for table in schema.keys() if not table.endswith('_sample')]
                    st.success(f"Database loaded successfully with {len(tables)} tables.")
                    
                    # Display tables in the sidebar
                    st.markdown("## Database Schema")
                    for table_name, columns in schema.items():
                        if not table_name.endswith('_sample'):
                            with st.expander(f"Table: {table_name}"):
                                for col_name, col_type in columns:
                                    st.text(f"• {col_name} ({col_type})")
                
                except Exception as e:
                    st.error(f"Error accessing database: {str(e)}")
                    st.info("Tip: Make sure the file is a valid SQLite database.")
        else:
            st.info("Creating a sample database for you to get started...")
            # Force refresh by just showing a button that reloads the page
            if st.button("Refresh Database List"):
                pass  # This will refresh the page when clicked
        
        # File upload section
        st.markdown("---")
        st.markdown("## Upload Database")
        
        # Create two columns - one for upload widget, one for a refresh button
        col1, col2 = st.columns([3, 1])
        
        with col1:
            uploaded_file = st.file_uploader("Upload a SQLite Database:", type=["db", "sqlite", "sqlite3", "db3"])
        
        with col2:
            refresh_button = st.button("🔄")
        
        # Handle file upload without using rerun()
        if uploaded_file is not None:
            # Check if this is a new upload by keeping track in session state
            current_upload = uploaded_file.name
            previous_upload = st.session_state.get('previous_upload', None)
            
            if current_upload != previous_upload:
                # Save the uploaded file to the databases directory
                file_path = os.path.join(db_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Database '{uploaded_file.name}' uploaded successfully!")
                
                # Set as current database
                st.session_state['current_db'] = file_path
                # Store this upload to prevent duplicate processing
                st.session_state['previous_upload'] = current_upload
                # Set flag for UI update
                st.session_state['db_just_uploaded'] = True
        
        # If a database was just uploaded, display info about it
        if st.session_state.get('db_just_uploaded', False):
            st.session_state['db_just_uploaded'] = False  # Reset the flag
            if 'current_db' in st.session_state and st.session_state['current_db']:
                try:
                    db_path = st.session_state['current_db']
                    schema = get_db_schema(db_path)
                    tables = [table for table in schema.keys() if not table.endswith('_sample')]
                    st.success(f"Database loaded with {len(tables)} tables. Ready to query!")
                except Exception as e:
                    st.error(f"Error accessing uploaded database: {str(e)}")
        
        # Clear history button
        st.markdown("---")
        if st.button("🗑️ Clear Query History"):
            st.session_state['history'] = []
            st.success("Query history cleared.")

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # App header
        st.markdown("<h1 style='text-align: center; color: white;'>UniSQLBot</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #CCCCFF; font-style: italic;'>Bridge between Natural Language and SQL Queries</p>", unsafe_allow_html=True)
        
        # Query input
        st.markdown("### Ask a question in plain English:")
        query = st.text_area("", 
                            value=st.session_state.get('rerun_query', ''),
                            height=100, 
                            placeholder="e.g., Show me all car makers from Germany",
                            label_visibility="collapsed")
        
        # Execute button
        if st.button("Generate SQL & Execute", type="primary", use_container_width=True):
            if not query:
                st.warning("Please enter a question to analyze.")
            elif 'api_key' not in st.session_state or not st.session_state['api_key']:
                st.warning("Please enter your OpenAI API key in the sidebar.")
            elif 'current_db' not in st.session_state or not st.session_state['current_db']:
                st.warning("Please select a database from the sidebar.")
            else:
                # Check if it's a general chat query
                if check_is_chat_query(query):
                    response = handle_chat_query(query)
                    st.markdown(f"<div style='background-color:#2D2D2D;padding:20px;border-radius:10px;margin-top:20px;color:#FFFFFF;'>{response}</div>", unsafe_allow_html=True)
                    
                    # Add to history
                    query_summary = {
                        'query': query,
                        'sql_query': "N/A - Conversational query",
                        'output': response,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'success': True
                    }
                    st.session_state['history'].append(query_summary)
                    st.session_state['rerun_query'] = None
                else:
                    with st.spinner("Converting natural language to SQL..."):
                        progress_text = st.empty()
                        progress_text.text("Step 1/3: Processing your query...")
                        
                        schema = get_db_schema(st.session_state['current_db'])
                        llm_response = process_natural_language_query(query, st.session_state['current_db'], schema)
                        
                        progress_text.text("Step 2/3: Extracting SQL query...")
                        sql_query = extract_sql_query(llm_response)
                        
                        progress_text.text("Step 3/3: Executing SQL query...")
                        result = execute_sql_query(sql_query, st.session_state['current_db'])
                        progress_text.empty()
                        st.session_state['rerun_query'] = None

                        st.markdown("<h3>Query Results</h3>", unsafe_allow_html=True)
                        
                        # Display the generated SQL query
                        st.markdown("#### Generated SQL Query")
                        st.code(sql_query, language="sql")
                        
                        # Display results
                        st.markdown("<div style='background-color:#2D2D2D;padding:20px;border-radius:10px;margin-top:20px;color:#FFFFFF;'>", unsafe_allow_html=True)
                        if result['success']:
                            query_summary = {
                                'query': query,
                                'sql_query': sql_query,
                                'data': result.get('data', pd.DataFrame()),
                                'output': result.get('output', ''),
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'figure': result.get('figure', None),
                                'success': True
                            }
                            st.session_state['history'].append(query_summary)

                            st.markdown("#### Results")
                            if 'data' in result and not result['data'].empty:
                                # Set dataframe styling for dark mode
                                st.markdown("""
                                <style>
                                .dataframe {
                                    background-color: #3D3D3D !important;
                                    color: #FFFFFF !important;
                                }
                                .dataframe th {
                                    background-color: #4D4D4D !important;
                                    color: #FFFFFF !important;
                                }
                                .dataframe td {
                                    background-color: #3D3D3D !important;
                                    color: #FFFFFF !important;
                                    border-color: #555555 !important;
                                }
                                </style>
                                """, unsafe_allow_html=True)
                                st.dataframe(result['data'], use_container_width=True)
                                
                                # Show row count
                                st.success(f"Found {result.get('row_count', len(result['data']))} results.")
                                
                                # If the DataFrame is empty but query was successful, show a more informative message
                                if len(result['data']) == 0:
                                    st.info("Query executed successfully, but no matching data was found.")
                            else:
                                st.code(result['output'], language=None)
                            
                            if result.get('figure'):
                                st.image(result['figure'], caption="Data Visualization", use_column_width=True)
                        else:
                            error_msg = result['error']
                            st.error(f"Error executing SQL query: {error_msg}")
                            # Add helpful suggestions based on common errors
                            if "no such table" in error_msg.lower():
                                tables = list(get_db_schema(st.session_state['current_db']).keys())
                                tables = [t for t in tables if not t.endswith('_sample')]
                                st.warning(f"Available tables in this database: {', '.join(tables)}")
                            elif "no such column" in error_msg.lower():
                                st.warning("Double-check column names in your SQL query.")
                                
                            # Still save the error in history
                            query_summary = {
                                'query': query,
                                'sql_query': sql_query,
                                'output': f"Error: {error_msg}",
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'success': False
                            }
                            st.session_state['history'].append(query_summary)
                        
                        st.markdown("</div>", unsafe_allow_html=True)

        # Conversation history
        st.markdown("<h3>💬 Conversation History</h3>", unsafe_allow_html=True)
        for i, entry in enumerate(st.session_state['history']):
            with st.container():
                st.markdown(f"""
                <div style='background-color:#2D2D2D;padding:15px;border-radius:10px;margin-bottom:10px;color:#FFFFFF;'>
                <b style='color:#9D8CFF;'>🧠 Question:</b> {entry['query']}<br>
                <b style='color:#9D8CFF;'>🕒 Time:</b> {entry['timestamp']}<br>
                <b style='color:#9D8CFF;'>💻 SQL Query:</b> <code style='background-color:#3D3D3D;color:#00FF00;'>{entry['sql_query']}</code><br>
                </div>
                """, unsafe_allow_html=True)
                
                if 'data' in entry and not entry.get('data', pd.DataFrame()).empty:
                    st.dataframe(entry['data'], use_container_width=True)
                elif 'output' in entry and 'success' in entry and entry['success']:
                    st.code(entry['output'], language=None)
                
                if entry.get('figure'):
                    st.image(entry['figure'], caption="📊 Visualization")

if __name__ == "__main__":
    main()