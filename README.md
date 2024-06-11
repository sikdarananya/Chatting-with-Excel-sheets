# Chatting-with-Excel-sheets

# Query Pipeline over Pandas DataFrames
This is a simple example that builds a query pipeline that can perform structured operations over a Pandas DataFrame to satisfy a user query, using LLMs to infer the set of operations.

This can be treated as the "from-scratch" version of our PandasQueryEngine.

# Define Modules
Here we define the set of modules:

Pandas prompt to infer pandas instructions from user query
Pandas output parser to execute pandas instructions on dataframe, get back dataframe
Response synthesis prompt to synthesize a final response given the dataframe
LLM
The pandas output parser specifically is designed to safely execute Python code. It includes a lot of safety checks that may be annoying to write from scratch. This includes only importing from a set of approved modules (e.g. no modules that would alter the file system like os), and also making sure that no private/dunder methods are being called.

# Build Query Pipeline
Looks like this: input query_str -> pandas_prompt -> llm1 -> pandas_output_parser -> response_synthesis_prompt -> llm2

Additional connections to response_synthesis_prompt: llm1 -> pandas_instructions, and pandas_output_parser -> pandas_output.
