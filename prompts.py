from langchain_core.prompts import ChatPromptTemplate


# Prompt templates
DOCUMENT_ANALYSIS_PROMPT = ChatPromptTemplate.from_template("""
You are an expert data analyst. Analyze these documents:

DOCUMENTS:
{documents}

INSTRUCTIONS:
1. Identify key information and patterns
2. Provide a detailed summary
3. Format in markdown with headings

ANALYSIS:
""")

DIRECT_ANSWER_PROMPT = ChatPromptTemplate.from_template("""
Answer the question based on your knowledge.

QUESTION:
{question}

If you don't know, say "I don't know" - don't make up answers.
Format your response in markdown.

ANSWER:
""")

WEB_SEARCH_PROMPT = ChatPromptTemplate.from_template("""
Answer using these web results:

QUESTION: {question}

RESULTS:
{search_results}

INSTRUCTIONS:
1. Synthesize the information
2. Cite sources when possible
3. If results don't help, say so

ANSWER:
""")