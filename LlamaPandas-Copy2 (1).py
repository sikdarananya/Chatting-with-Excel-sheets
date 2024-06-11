#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import openai
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.callbacks.base import BaseCallbackHandler
from threading import Thread
from queue import Queue, Empty
from collections.abc import Generator
from sse_starlette.sse import EventSourceResponse
import json
from llama_index.core import StorageContext
from llama_index.llms.openai import OpenAI
from IPython.display import Markdown, display
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
)

from llama_index.experimental.query_engine.pandas import (
    PandasInstructionParser,
)
from llama_index.core import PromptTemplate
from llama_index.core import ServiceContext

import logging
import sys

logging.basicConfig(
stream = sys.stdout, level=logging.INFO)

logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# In[2]:


api_key = "3289261e6cc84fa8aef58d38e2264fa9"
openai.api_key = api_key
openai.api_base = 'https://openai-demo-mb-001.openai.azure.com/'
openai.api_type = 'azure'
openai.api_version = '2023-05-15'
deployment_name = 'openaidemomb001'
deployment_name_embeddings = 'openaidemomb002'

os.environ["OPENAI_API_TYPE"] = openai.api_type
os.environ["OPENAI_API_VERSION"] = openai.api_version
os.environ["OPENAI_OPENAI_BASE"] = openai.api_base
os.environ["OPENAI_API_KEY"] = "3289261e6cc84fa8aef58d38e2264fa9"


# In[3]:


llm = AzureOpenAI(
    deployment_name = deployment_name,
    model_name = deployment_name,
    cache = False,
    api_key = api_key,
    azure_endpoint = openai.api_base,
    temperature = 0.1)


# In[4]:


import pandas as pd

df = pd.read_excel('Wyndham_Managed_services_6M_Inflow.xlsx', '6M_Inflow')
df.head(50)


# In[5]:


table_desc_str = (
    "The following are the descriptions of the columns in the dataframe\n"
    "1. Number: A unique identifier automatically assigned to each incident.\n"
    "2. Priority: The level of importance and urgency for this incident. Options include 1 - Critical, 2 - High, 3-Moderate, 4-Low and SR.\n"
    "3. State: The current status of the incident from options like Active, Asssigned, Awaiting Change, Awaiting Release, Awaiting User Info, Awaiting Vendor, Closed, New, Pended Per Business Needs, Resolved.\n"
    "4. Assignment group: The team or group responsible for resolving the incident from the list provided. The list includes RCI Application Support L1, RCI B2B, RCI B2C, RCI Back Office, RCI Call Center, RCI Internal IT, RCI Job Failure, RCI Points Spt, RCI TRC Clubs, RCI Weeks App Support, WVC Application L1 Support, WVC ETL L2 Support, WVC Focus L2 Support, WVC Fusion L2 Support, WVC IRIS L2 Support, WVC Payment Gateway L2 Support, WVC Sales Marketing L2 Support, WVC Voyager L2 Support, WYND Data Svcs ETL\n"
    "5. Assigned to: The individual assigned to handle this incident from the available personnel within the assignment group.\n"
    "6. Description: A detailed explanation of the incident, including what happened, how it was discovered, and any steps already taken to address it.\n"
    "7. Close code: The reason for closing the incident. Options might include 3rd Party Issue, 3rd Party/ Vendor Change, 3rd Party/ Vendor Issue, Account Disable, Account Management, Account Create, App Issue, App Support, Alert Cleared, Application Error, Auto Resolved, Application Issue, Certificate Installation, Backup/Restore, Backed Out Change, Closed/Resolved by Caller, Code Deployment, Config Change, Config Issue, Connectivity Issue, Configuration Issue, Data Issue, Data Analysis, Data Setup, Data Change, CPU Utilization, Data Setup, Database Error, Db Patching, Db Locks, Duplicate Event, Decommission, Disk Space, Disk Issue, FTP Failure, File Restored, File Removed, Install, ISP/Carrier Outage, Hardware Error, Hardware/Infrastructure Issue, Job Failure, Known Error, Long Runng Job, License Error, etc.\n"
    "8. Close notes: Any additional comments or details about the closure of the incident. Including any follow-up actions or relevant information.\n"
    "9. Opened: The date and time when the incident was first reported. Consider the format as YYYT-MM-DD. Example : 2024-01-01.\n"
    "10. Resolved: Date and time when the incident was resolved.Consider the format as YYYY-MM-DD. Example : 2024-02-02.\n"
    "11. Resolved by: The name of the individual who resolved the incident from the list of personnel.\n"
    "12. Severity: The impact level of the incident on the business. Options include 3-Low.\n"
    "13. Closed: The date and time when the incident was officially closed Consider the format as YYYY-MM-DD. Example : 2024-02-02.\n"
    "14. Closed by: The name of the individual who closed the incident from the list of personnel.\n"
    "15. Configuration item: The specific asset or service affected by the incident from the list provided.\n"
    "16. Incident Type: The category of the incident. Options include Proactive and Reactive.\n"
    "17. Category: A more detailed classification of the incident within the chosen type from the list. List includes Incident, Informational and Request.\n"
    "18. Impact: The extent to which the incident affects the business. Options include 1-Extensive/Widespread, 2-Significant/Large, 3-Moderate/Limited, 4-Minor/Localized.\n"
    "19. Urgency: How quickly the incident needs to be resolved. Options include 1-Critical, 2-High, 3-Medium, 4-Low.\n"
    
)

sample_query = (
   """query_str: Share the count of all incidents whose Impact is Significant/Large in the month of January 2024.
               pandas_instructions: assistant: df[(df['Impact']=='2-Significant/Large') & (df['Opened'].dt.month==1) & (df['Opened'].dt.year==2024)].shape[0]\n"""
    
    """query_str : Create a bar chart based on incident numbers and Incident Type.
       pandas_instructions = df.groupby('Incident Type')['Number'].count().plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon'])
    """
    """
       query_str = Create a trend chart based on incident numbers over time.
       pandas_instructions = df.plot(f, x_col, y_col, title='Trend Graph', xlabel='Time', ylabel='Values', marker='o', linestyle='-', color='lightgreen', figsize=(10, 6), grid=True, trendline=True
    """
    """
       query_str = Create a line graph 
       pandas_instructions = df.plot(df, x_col, y_col, title='Line Graph', xlabel='X-axis', ylabel='Y-axis', marker='o', linestyle='-', color='lightgreen', figsize=(10, 6), grid=True)
    """
    """
      query_str = Create a Bar chart
      pandas_instructions = df.plot(df, x_col, y_col, title='Bar Chart', xlabel='X-axis', ylabel='Y-axis', color='lightblue', figsize=(10, 6), grid=True)
    """
    """
       query_str = Create a pie chart
       pandas_instructions = df.plot(df, labels_col, values_col, title='Pie Chart', colors=None, explode=None, autopct='%1.1f%%', startangle=90, figsize=(8, 8))
       
    """
    """
       query_str = Create a stacked bar chart
       pandas_instructions = df.plot(df, x_col, y_cols, title='Stacked Bar Chart', xlabel='X-axis', ylabel='Y-axis', colors=None, figsize=(10, 6), grid=True)
    """

    """
       query_str: Count the number of incidents which have the same Urgency as the ticket number INC2383678 in the month of January 2024
       pandas_instructions: assistant: df[(df['Urgency'] == df[df['Number'] == 'INC2383678']['Urgency'].iloc[0]) & (df['Opened'].dt.month == 1) & (df['Opened'].dt.year == 2024)]['Number'].count()
    """
    """
       query_str: How many incidents have the same Urgency as the ticket above in the month of February 2024?
       pandas_instructions: assistant: df[df['Urgency'] == '3-Medium'][df['Opened'].dt.month == 2][df['Opened'].dt.year == 2024]['Number'].count()
    """
)



instruction_str = (
    "1. Convert the query to executable Python code using Pandas.\n"
    "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
    "3. The code should represent a solution to the query.\n"
    "4. Please follow the {table_desc_str} for reference while creating the python code from the query.\n"
    "5. PRINT ONLY THE EXPRESSION.\n"
    "6. Do not quote the expression.\n"
    "7. Follow {sample_query} format.\n"
    "8. Last months refer to the months previous to the current date, including the current date."
    
    
)

pandas_prompt_str = (
    "You are working with a pandas dataframe in Python.\n"
    "The name of the dataframe is `df`.\n"
    "This is the result of `print(df.head())`:\n"
    "{df_str}\n\n"
    "Follow these instructions:\n"
    "{instruction_str}\n"
    "Query: {query_str}\n\n"
    "Expression:"
)
response_synthesis_prompt_str = (
    "Given an input question, synthesize a response from the query results.\n"
    "Query: {query_str}\n\n"
    "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
    "Pandas Output: {pandas_output}\n\n"
    "Response: "
)

pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
    instruction_str=instruction_str, df_str=df.head(10)
)
pandas_output_parser = PandasInstructionParser(df)
response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)
llm = llm


# In[6]:


from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer

# Initialize ChatMemoryBuffer
memory = ChatMemoryBuffer.from_defaults(token_limit=1500)


# In[7]:


qp = QP(
    modules={
        "input": InputComponent(),
        "pandas_prompt": pandas_prompt,
        "llm1": llm,
        "pandas_output_parser": pandas_output_parser,
        "response_synthesis_prompt": response_synthesis_prompt,
        "llm2": llm,
    },
    verbose=True,
    memory=memory  # Include memory in the Query Pipeline
)

# Add the conversational memory module as a chain
qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])

# Add links for processing the input and generating the response
qp.add_links(
    [
        Link("input", "response_synthesis_prompt", dest_key="query_str"),
        Link("llm1", "response_synthesis_prompt", dest_key="pandas_instructions"),
        Link("pandas_output_parser", "response_synthesis_prompt", dest_key="pandas_output"),
    ]
)

# Add link from response synthesis prompt to llm2
qp.add_link("response_synthesis_prompt", "llm2")


# In[29]:


response = qp.run(
    query_str= "Who closed INC2383678?",
)


# In[30]:


print(response.message.content)


# In[31]:


response = qp.run(
    query_str= "Give me details of the ticket above",
)

print(response.message.content)


# In[34]:


response = qp.run(
    query_str= "Which month was the above ticket opened?",
)

print(response.message.content)


# In[35]:


response = qp.run(
    query_str= "What is the Urgency of the above ticket?",
)

print(response.message.content)


# In[36]:


response = qp.run(
    query_str= "How many tickets have the same Urgency as the ticket above?",
)

print(response.message.content)


# In[38]:


response = qp.run(
    query_str= "How many tickets have the same Urgency as the ticket above in the month of January 2024?",
)

print(response.message.content)


# In[8]:


response = qp.run(
    query_str= "What are the details of incident number INC2383678?",
)

print(response.message.content)


# In[9]:


response = qp.run(
    query_str= "How many incidents have the same Urgency as the ticket above in the month of February 2024?",
)

print(response.message.content)


# In[14]:


response = qp.run(
    query_str= "How many incidents have the same Urgency as the ticket above in the month of January 2024?",
)

print(response.message.content)


# In[15]:


response = qp.run(
    query_str= "What is the incident number of the ticket which was opened at the very beginning of January 2024 that has the same Urgency as the ticket above?",
)

print(response.message.content)


# In[16]:


response = qp.run(
    query_str= "When was incident number INC2428260 opened and what is it's Urgency?",
)

print(response.message.content)


# In[17]:


response = qp.run(
    query_str= "Which incident was opened on 1st January, 2024?",
)

print(response.message.content)


# In[8]:


response = qp.run(
    query_str= "What is the State of incident number INC2383678?",
)

print(response.message.content)


# In[11]:


response = qp.run(
    query_str= "Also tell me about the Urgency of the above incident number?",
)

print(response.message.content)


# In[8]:


response = qp.run(
    query_str= "Provide me the trend for the number incident numbers with FOCUS in their descriptions and Impact 2-Significant/Large over the last 3 months. Also prepare a bar chart for the result",
)

print(response.message.content)


# In[29]:


response = qp.run(
    query_str= "Provide me with the incident numbers with FOCUS in their Descriptions and Impact as Significant/Large over the last 3 months. ",
)

print(response.message.content)


# In[13]:


response = qp.run(
    query_str= "When was incident numbers INC2491616 and INC2519259 opened? ",
)

print(response.message.content)


# In[22]:


response = qp.run(
    query_str= "What is the Impact of the above incident numbers INC2491616 and INC2519259?",
)

print(response.message.content)


# In[9]:


response = qp.run(
    query_str= "Count the number of incident numbers with Focus or FOCUS application in the months of February, March and April 2024?",
)

print(response.message.content)


# In[11]:


response = qp.run(
    query_str= "Draw a stacked bar chart based on the above result with respect to their Impact and their Opening month? Show months name in X axis in order for only the months of January, February and March in 2024",
)

print(response.message.content)


# In[13]:


response = qp.run(
    query_str= "Design a Trend graph for all incidents numbers in the year 2023 based on their Impact",
)

print(response.message.content)


# In[14]:


response = qp.run(
    query_str= "Design a line graph for all incidents numbers in the year 2023 based on their Impact",
)

print(response.message.content)


# In[15]:


response = qp.run(
    query_str= "Design a pie chart for all incidents numbers in the year 2023 based on their Impact",
)

print(response.message.content)


# In[16]:


response = qp.run(
    query_str= "Design a Bar graph for all incidents numbers in the year 2023 based on their Impact",
)

print(response.message.content)


# In[ ]:




