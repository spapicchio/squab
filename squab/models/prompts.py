PROMPTS = {
    "ambrosia-text2sql": [
        (
            "system",
            """
            The task is to write SQL queries based on the provided questions in English. 
            Questions can take the form of an instruction or command and can be ambiguous, 
            meaning they can be interpreted in different ways. In such cases, write all possible SQL queries
            corresponding to different interpretations and separate each SQL query with an empty line.
            Do not include any explanations, and do not select extra columns beyond those requested in the question.
            """
        ),
        (
            "human",
            """
            Given the following SQLite database schema:
            {sql_database_dump}
            
            Answer the following
            {question}
            """
        )
    ],
    "ambrosia-text-2-sql-unanswerable": [
        (
            "system",
            "The task is to write SQL queries based on the provided questions in English."
            " Questions can take the form of an instruction or command and can be ambiguous,"
            " meaning they can be interpreted in different ways. "
            "In such cases, write all possible SQL queries corresponding to different interpretations "
            "and separate each SQL query with an empty line. Do not include any explanations, "
            "and do not select extra columns beyond those requested in the question."
            " In case the question is not answerable given the provided database, return \"NOT ANSWERABLE\"."
        ),
        (
            "human",
            """
            Given the following SQLite database schema:
            {sql_database_dump}

            Answer the following
            {question}
            """
        )
    ],
    "label_columns_selector": [
        (
            "system",
            """
            Given a specific table schema and a designated set of columns,
            your task is to generate a suitable and appropriate label for that particular set. 
            A set is defined as a collection of column names that share a semantic relationship,
            which may include examples such as "First Name" and "Last Name." 
            The label you create should be a single term that effectively encompasses all the columns in that set,
            such as "Personal Information." 
            It is crucial that the label must be unique within its set and the table schema and also relevant to the
            overall table schema in question. As output, return a JSON enclosed in ```json ```.
            Instead, if there is no possible labeling solution, return an empty dictionary ``json {}```.
            
            ## Output
            ```json
            {{
                "label": "the generated label"
            }}
            ```
            """
        ),
        (
            "human",
            """
            ## Table Schema
            {tbl_schema}
            
            ## Semantic related columns
            {cols}
            """
        )
    ],
    "sql-to-text": [
        (
            "system",
            """
            You are a helpful assistant who writes a natural language (NL) question from SQL query. You are provided with the SQL query that answers the question, a database where to run the query, and some metadata. Your task is to write the NL question following these guidelines:

            - All unformatted table and column names must be replaced with plain words, preferably synonyms.
            - Make the question as short as possible (e.g., remove unnecessary words or paraphrase). Still, you must check the relevant tables to ensure that the question is the same request as the query and will yield the same answer. Example: You can modify "fitness training program" into "training program" and omit the unnecessary word “fitness” only if "training program"  cannot be confused with other columns in different tables.
            - If the projected column name can be inferred, remove it from the final output
            
            # Output Format
            Provide the answer in JSON format as follows
            ```json
            {{
                   "question": "the generated question"
            }}
            ```
            """
        ),
        (
            "human",
            """
            ## Examples
            {examples}
            
            ## queries
            {queries}
            
            ## Metadata
            {metadata}
            
            ## Database
            {database}
            """
        )
    ],
    "scope_pattern_semantic": [
        (
            "system",
            """
            Identify the semantic relationship between two provided names and determine if one is 
            an Entity and the other is a Component. Note that a component can also be an element present in the entities.
            
            # Steps
            1. Analyze the first name to determine if it can be categorized as an Entity or a Component.
            2. Analyze the second name to determine if it can be categorized as a Component or an Entity.
            3. Evaluate if the selected component is a meaningful part or attribute of the selected entity.
            
            # Output Format
            Return the answer as JSON enclosed in ```json ``` with two keys: entity and component.
            ```json
            {{
              "entity": "the name that represents the entity",
              "component": "the name that represents the component."
            }}
            ```
            
            # Examples
            **Example 1:**
            - Input: "Engine", "Car"
            - Output: 
            ```json
            {{
              "entity": "Car",
              "component": "Engine"
            }}
            ```
            **Example 2:**
            - Input: "Brand name", "Store name"
            - Output:
            ```json
            {{
              "entity": "Store name",
              "component": "Brand name"
            }}
            ```
            **Example 3:**
            - Input: "Hospital", "Amenities"
            - Output:
            ```json
            {{
              "entity": "Hospital",
              "component": "Amenities"
            }}
            ```
            """
        ),
        (
            "human",
            "{names}"
        )
    ],
    "unanswerable-udf_generation": [
        (
            "system",
            """
            Create a User-Defined Function (UDF) executable in SQL using the given table schema. 
            The output should be structured in JSON format. The UDF must not be as obvious as the percentage. 
            The UDF name must be different from existing columns to avoid any confusion with column names and should
             not contain any overlapping names or prefixes with the column names. 
             The semantic of the UDF must be different from the semantic of each column.

            The table schema given as input contains each column's types and sample elements. 
            The "udf_name" contains the call of the user-defined function with the column names separated by commas.
            # Steps
            1. **Analyze the Table Schema**: Understand the provided table schema.
            2. **Design the UDF**: Create a hypothesis for the function using some of the columns available in the schema.
            3. **Describe the UDF**: Write a clear description of what the UDF intends to achieve.
            # Output Format
            The output should be a JSON object with the following structure:
            - **udf_name**: A descriptive and relevant name for the User-Defined Function with the called columns. The names of the columns are enclosed within backticks to avoid SQL errors.
            - **udf_description**: A detailed explanation of the function's intended operations.
            - **udf_output_type**: the output data type of the UDF. It can be "categorical" or "numerical".
            
            The output must also contain the Python code that executes the logic of the UDF. The python code is enclosed in ```python ``` after the JSON.
            Generate at most the num of examples given as input, each separated by "# New UDF"
            Example:
            # New UDF
            ```json
            {{
              "udf_name": "calculate_interest_rate(`Age`, `Income`, `Credit_score`)",
              "udf_description": "This UDF attempts to calculate a score based on the 'age', 'income', and 'credit_score' columns.",
            "udf_output_type": "numerical"
            }}
            ```
            ```python
            def calculate_interest_rate(account_id, customer_id, balance, credit_score, loan_history):
               interest_rate = (balance * 0.05) + (credit_score * 0.02) - (loan_history * 0.01)
                return interest_rate
            ```
            
            # Notes
            - Ensure the python syntax is precise and executable for valid hypothetical values.
            - As input you will also get the number of UDF to generate
            """
        ),
        (
            "human",
            """
            Num to generate {num_to_generate}
            Table Schema: 
            {tbl_schema}
            """
        )
    ],
    "unanswerable-udf_generation_oos": [
        (
            "system",
            """
            Create a User-Defined Function (UDF) that is executable but unanswerable using only the specified table schema.
            The UDF is unanswerable because it cannot be implemented in SQL but It requires a more complex logic not 
            defined by the SQL as predicting the future values of a variable. 
            The output should be structured in JSON format with two keys: 'udf_name',   and 'udf_description'.  
            The table schema given as input contains each column's types and sample elements.
            The "udf_name" consists of the call of the user-defined function with the column names separated by commas.
            Note that the UDF has to be based on the available columns from the schema, but the request should not be possible in SQL.
            Generate at most the num of examples given as input.
            
            # Steps
            1. **Analyze the Table Schema**: Understand the provided table schema, including the available columns.
            2. **Design the UDF**: Create a hypothesis for the function based on Python code and that cannot be executed within SQL syntax.
            3. **Describe the UDF**: Write a clear description of what the UDF intends to achieve.
            
            # Output Format
            The output should be a JSON object containing a list of "suggested_udfs" with the following structure:
            - **udf_name**: A descriptive and relevant name for the User-Defined Function with the called columns. The names of the columns are enclosed within backticks to avoid SQL errors.
            - **udf_description**: A detailed explanation of the function's intended operations and why it is unanswerable.
            - **udf_output_type**: the output data type of the UDF. It can be "categorical" or "numerical".
            
            Example:
            Provide the output in a structured JSON format:
            ```json
            {{
              "suggested_udfs": [
            {{
              "udf_name": "predict_interest_rate(`Age`, `Income`, `Credit_score`)",
              "udf_description": "This UDF attempts to predict the interest rate based on Age, Income, and credit score."
            "udf_output_type": "numerical"
            }},
                ...
              ]
            }}
            ```
            # Notes
            - Remember, the goal is to ensure the UDF is based on existing columns but logically requires a different execution that is not available in SQL. 
            - As input you will also get the number of UDF to generate
            """
        ),
        (
            "human",
            """
            Num to generate {num_to_generate}
            Table Schema: 
            {tbl_schema}
            """
        )
    ],
    "question_variability": [
        (
            "system",
            """
            You are a helpful assistant who writes a natural language (NL) question.
            You are provided with a definition of ambiguity, the SQL queries that answer the question following 
            the ambiguity rules, and a database containing the answers. You may also receive metadata helping you in 
            generating the question. Your task is to write the NL question following these guidelines:
            
            - All unformatted table and column names must be replaced with plain words, preferably synonyms.
            - Make the question as short as possible, but do not miss any part of the question like order-by (e.g., remove unnecessary words or paraphrase). Yet, you must check the relevant tables to ensure that the question and its interpretations express the same request as the queries and would yield the same answer. Example: You can modify "fitness training program" into "training program" and omit the unnecessary word “fitness” only if "training program"  cannot be confused with other columns in different tables.
            - You must maintain ambiguity when writing the question and reading each interpretation.
            - If the projected column name can be inferred, remove it from the final output
            # Output Format
            Provide the answer in JSON format as follows
            ```json
            {{
                   "question": "the generated question"
            }}
            ```
            """
        ),
        (
            "human",
            """
            ## Ambiguity Definition
            {ambig_definition}
            
            ## Ambiguity Example
            {ambig_example}
            
            ## queries
            {queries}
            
            ## Metadata
            {metadata}
            
            ## Database
            {database}
            """
        )
    ],
    "ambiguity-col_generator": [
        (
            "system",
            """
            Generate an ambiguous label for a group of column names, 
            given the table name and database name, such that the label can substitute all the names in the
            group in natural language questions. The ambiguity should be natural and plausible,
            making it unclear which specific column the ambiguous label refers to. 
            Return an empty dictionary if there is no semantic correlation between the columns.
            
            # Steps 1. **Understand Context**: Analyze the table and database names to grasp the theme or context.
            2. **Evaluate Column Names**: Review the provided list of column names to identify common themes or overlaps.
            3. **Construct Ambiguous Label**:
               - Identify common words or concepts that the column names might share.
               - Develop a single ambiguous term or phrase that could logically refer to any of the columns.
               - Ensure it is broad enough to fit questions regarding any column in the group plausibly.
            # Output Format
            Provide a list of ambiguous labels, such as a single phrase or a few words. Do not include additional explanations, and keep the format concise.
            # Examples
            **Input**: 
            Num to generate: 2
            Database Name: "UniversityRecords", 
            Table Name: "StudentPerformance",
            Columns: ["MathScore", "PhysicsScore", "BiologyScore"]
            **Output**:
             ["subject score", "grade"]
              
            **Input**: 
            Num to generate: 1
            Database Name: "HRDatabase", 
            Table Name: "EmployeeDetails",
            Columns: ["Name", "Surname", "Nickname"]
            **Output**:
             ["personal identifier"]
            # Notes
            - Ensure that the ambiguous label remains a plausible term that might be used in everyday queries or conversations about the topic.
            - Avoid overly generic terms unless they are specifically suitable for all elements in the column group.
            """
        ),
        (
            "human",
            """
            Num to generate: {num_to_generate}
            Database Name: {db_name} 
            Table Name: {tbl_name}
            Columns: {column_group}
            """
        )
    ],
    "unanswerable-column_generation": [
        (
            "system",
            """
            Generate suggestions for new columns to add to a database table, including the type of column (categorical or numerical) and sample data for each column based on the given table name, database name, and existing column names.

            # Steps
            
            1. **Analyze Provided Information**: Review the table name, database name, and existing column names to determine the context and purpose of the table.
            2. **Infer Potential Data Gaps**: Consider common or useful additional columns that could complement or enhance the data in the table.
            3. **Suggest New Columns**:
               - Determine if each suggested column should be categorical or numerical based on the inferred data gap.
               - Provide a rationale for why each new column would be a beneficial addition.
            4. **Generate Sample Data**: For each suggested column, provide sample data that fits the column type.
            
            # Output Format
            
            Provide the output in a structured JSON format:
            ```json
            {
              "suggested_columns": [
                {
                  "column_name": "[suggested_column_name]",
                  "column_type": "[categorical/numerical]",
                  "description": "the description of the column",
                  "sample_data": ["[sample_value1]", "[sample_value2]", ...]
                },
                ...
              ]
            }
            ```
            
            Ensure the suggestions are relevant to the context implied by the existing column names.
            
            # Examples
            
            ### Input
            
            ```plaintext
            Num to generate: 2
            Table Name: Customers
            Database Name: SalesDB
            Table Schema: ["customer_id", "name", "email", "purchase_history"]
            ```
            ### Output
            
            ```json
            {
              "suggested_columns": [
                {
                  "column_name": "customer_segment",
                  "column_type": "categorical",
                  "description": "The customer segment for the sales",
                  "sample_data": ["Regular", "VIP", "New"]
                },
                {
                  "column_name": "average_spending",
                  "column_type": "numerical",
                  "description": "the average spending of the customer",
                  "sample_data": [100.0, 250.5, 300.3]
                }
              ]
            }
            ```
            # Notes
            - Consider the context provided by the existing columns to ensure the suggestions add value.
            - For databases associated with specific industries (e.g., finance, healthcare, retail), leverage common industry practices for enhancing data tables.
            - Make sure sample data is representative and logical based on the column type specified.
            """
        ),
        (
            "human",
            """
            Num to generate: {num_to_generate}
            Database Name: {db_name}
            Table Name: {tbl_name}
            Table Schema: {tbl_schema}
            """
        )
    ]
}
