import gc               
import os
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
import io
import contextlib
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Set non-interactive backend for matplotlib
matplotlib.use('Agg')

# Load API keys from .env
load_dotenv()

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Streamlit UI
st.set_page_config(page_title="Excel Chatbot", layout="wide")
st.title("📊 Excel/CSV Chatbot")

# Upload file
uploaded_file = st.file_uploader("📂 Upload your Excel, XLS, or CSV file", type=["csv", "xlsx", "xls"])

if uploaded_file:
    # Load data
    filename = uploaded_file.name.lower()
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"❌ Could not read file: {e}")
        st.stop()

    # Show preview
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # Show columns
    # st.write("**🧩 Columns detected:**", list(df.columns))

    # User question
    user_query = st.text_input("💬 Ask a question about the data:")

    if user_query:
        with st.spinner("🤖 Thinking..."):
            # Build prompt for LLM with graph instructions
            prompt = f"""
You are a Python data expert.
Available DataFrame columns: {list(df.columns)}

just Write Python code to answer this question from the DataFrame `df`:
Question: "{user_query}"

⚠️ IMPORTANT RULES:
1. Only return Python code (no explanations or markdown)
2. Always store your final result in a variable named `result`
3. Use ONLY these imports: pandas as pd, numpy as np, matplotlib.pyplot as plt
4. Your code must work directly on the `df` variable
5. Handle missing values appropriately
6. If showing data, limit to 50 rows maximum
7. Give me the perticular raw or colummn(for example give the raw no 10/11/15 or any same goes with columns)
8. questions like : Show me/give me/ tell me about the column/row or conditions, then show/give that perticular the data for the column/column 
   - example : give/show/ tell me about the data for/of the row 45,4,6 then give the data for  row 45,4,6
   - for comperision columns make like give those perticular column data and there could be the multiple columns
      example : column(or name) vs column(names) vs column(names) then show data of columns and same with rows
      exception : "columns 1 vs 2":  result = df[['Pregnancies', 'Glucose']].head(100) these use the print statement  for to show the result and use "print(result)" and just write the code only
     do not use " Here is the Python code to answer the question:" 
```
result = df[['Pregnancies', 'Glucose', 'BloodPressure']].head(50)
```
      just write the code like " result = df[['Pregnancies', 'Glucose', 'BloodPressure']].head(50)
      print(result)"

9. For plots:
   - use the matplotlib for graph making not the seaborn
   - always put time on x-axis and use do not overlap names
   - Start with `plt.figure(figsize=(6, 3))` to create appropriately sized figures
   - Store the figure in `result`
   - Use `plt.close()` after creating plots to prevent display issues
   -use the plt for gragh not the other python lib
   - Add titles and labels for clarity
   - do not use the seaborn lib
   -use the different-different color with names for graphs for all graph
   - Exception for pie chart " import gc,gc.collect()" use this for pie graph
   - Exception for scatter plot and boxplot and bar graph do not use the " plt.close() ,result = plt.gcf() , just use the plt.show()   
"""

            try:
                # Get response from Groq API
                response = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {"role": "system", "content": "You are an expert Python data assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )
                generated_code = response.choices[0].message.content.strip()
                
                # Clean code output
                if generated_code.startswith('```python'):
                    generated_code = generated_code[9:-3].strip()
                elif generated_code.startswith('```'):
                    generated_code = generated_code[3:-3].strip()
                
                # st.subheader("🧠 Generated Code")
                # st.code(generated_code, language='python')

                # Prepare execution environment
                env = {
                    'df': df.copy(),
                    'pd': pd,
                    'np': np,
                    'plt': plt,
                    'sns': sns,
                    'gc': gc
                }
                
                # Capture stdout for print statements
                output_capture = io.StringIO()
                result = None
                
                try:
                    with contextlib.redirect_stdout(output_capture):
                        exec(generated_code, env)
                        result = env.get('result', None)
                except Exception as e:
                    st.error(f"🚨 Code execution error: {str(e)}")
                    st.stop()

                # Get captured output
                captured_output = output_capture.getvalue()
                
                st.subheader("✅ Results")
                
                # Display captured output if exists
                if captured_output:
                    st.write("**📝 Console Output:**")
                    st.code(captured_output)
                
                # Display result based on its type
                if result is not None:
                    # Handle plot objects
                    if isinstance(result, (matplotlib.figure.Figure, plt.Axes)):
                        st.subheader("📈 Visualization")
                        if isinstance(result, plt.Axes):
                            fig = result.get_figure()
                        else:
                            fig = result
                        fig.set_size_inches(6, 3)
                        st.pyplot(fig, use_container_width=False)
                    
                    # Handle dataframes
                    elif isinstance(result, pd.DataFrame):
                        if len(result) > 100:
                            st.warning(f"⚠️ Showing first 100 rows of {len(result)} total rows")
                        st.dataframe(result.head(100))
                    
                    # Handle series
                    elif isinstance(result, pd.Series):
                        st.dataframe(result.to_frame())
                    
                    # Handle scalar values
                    elif isinstance(result, (int, float, str)):
                        st.metric("Result", result)
                    
                    # Handle other types
                    else:
                        st.write(result)
                    
                    # Prepare data for insights summary
                    result_info = {
                        "user_query": user_query,
                        "result_type": type(result).__name__,
                        "result_preview": ""
                    }
                    
                    # Create preview based on result type
                    if isinstance(result, pd.DataFrame):
                        result_info["result_preview"] = result.head(5).to_string()
                    elif isinstance(result, pd.Series):
                        result_info["result_preview"] = result.head(10).to_string()
                    elif isinstance(result, (int, float, str)):
                        result_info["result_preview"] = str(result)
                    elif isinstance(result, (matplotlib.figure.Figure, plt.Axes)):
                        result_info["result_preview"] = "Visualization generated"
                    
                    # Call Groq for natural language summary
                    with st.spinner("💡 Generating insights..."):
                        groq_prompt = f"""
Based on the user's query: "{result_info['user_query']}"

We have a result of type: {result_info['result_type']}
Result preview:
{result_info['result_preview']}

use the minimum token for answer with all numerica data details
Please provide a concise natural language summary answering the user's original question.
Explain what the results mean in the context of their data.
do not make the graph just summarize the data for graph
"""
                        insights_response = client.chat.completions.create(
                            model="llama3-8b-8192",
                            messages=[
                                {"role": "system", "content": "You are an expert data analyst who explains results in simple terms. do not talk outside the data"},
                                {"role": "user", "content": groq_prompt}
                            ],
                            temperature=0.3,
                            max_tokens=200
                        )
                        insights_summary = insights_response.choices[0].message.content
                        
                        st.subheader("💡Summary")
                        st.write(insights_summary)
                        
                else:
                    st.info("💡 Code executed successfully but no 'result' variable was created")
                
            except Exception as e:
                st.error(f"❌ Error generating response: {e}")