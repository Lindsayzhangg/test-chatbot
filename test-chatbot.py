import streamlit as st
from langchain_voyageai import VoyageAIEmbeddings
import os
import boto3
from urllib.parse import urlparse
import pinecone
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, RetrievalQA
import time
import re
import json
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import uuid
import warnings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_pinecone import PineconeVectorStore
from langchain_voyageai import VoyageAIEmbeddings
from langchain.chains import create_history_aware_retriever
from datasets import Dataset
from ragas.metrics import context_relevancy, answer_relevancy, faithfulness, context_recall, answer_correctness
from ragas.metrics.critique import harmfulness
from ragas import evaluate
import pandas as pd
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import traceback

# Ignore all warnings
warnings.filterwarnings("ignore")

# Set up Streamlit app
st.set_page_config(page_title="Patient Chatbot", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f8ff;
        color: #2f4f4f;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
    }
    .stButton>button:hover,
    .stButton>button:active,
    .stButton>button:focus {
        background-color: #1e90ff; /* 将鼠标悬停和点击时的背景色改为蓝色 */
        color: white !important; /* 将鼠标悬停和点击时的文字颜色改为白色 */
    }
    .stTextInput>div>div>input {
        background-color: #f8f8f8;
        color: #2f4f4f;
        font-size: 16px;
        padding: 10px;
        border-radius: 8px;
        border: 2px solid #4CAF50;
    }
    .stTextInput>div>div>input:focus {
        border: 2px solid #1e90ff !重要; 
        box-shadow: none !重要; 
    }

    .stDownloadButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
    }
    .stDownloadButton>button:hover,
    .stDownloadButton>button:active,
    .stDownloadButton>button:focus {
        background-color: #1e90ff; 
        color: white !重要; /* 确保点击后文字颜色为白色 */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Patient Chatbot with Retrieval Abilities")

# Function to generate pre-signed URL
def generate_presigned_url(s3_client, s3_uri):
    parsed_url = urlparse(s3_uri)
    bucket_name = parsed_url.netloc
    object_key = parsed_url.path.lstrip('/')
    presigned_url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket_name, 'Key': object_key},
        ExpiresIn=3600
    )
    return presigned_url

# Function to retrieve documents, generate URLs, and format the response
def retrieve_and_format_response(user_input, retriever, llm_chain):
    response = llm_chain.invoke({"input": user_input}, config={"configurable": {"session_id": "test"}})
    return response

# New function to save retrieved documents to a file without sensitive information
def save_retrieved_docs_to_file(docs):
    retrieved_docs_content = []
    for doc in docs:
        retrieved_docs_content.append({
            "content": doc.page_content,
            "more_info": f"URL_PLACEHOLDER_{doc.metadata['id']}"
        })
    return json.dumps(retrieved_docs_content, indent=2)

# Function to dynamically replace URL placeholders with pre-signed URLs
def replace_placeholders_with_urls(content, s3_client):
    data = json.loads(content)
    for item in data:
        s3_uri = item["more_info"].replace("URL_PLACEHOLDER_", "")
        presigned_url = generate_presigned_url(s3_client, s3_uri)
        item["more_info"] = presigned_url
    return json.dumps(data, indent=2)

# Setup - Streamlit secrets
OPENAI_API_KEY = st.secrets["api_keys"]["OPENAI_API_KEY"]
VOYAGE_AI_API_KEY = st.secrets["api_keys"]["VOYAGE_AI_API_KEY"]
PINECONE_API_KEY = st.secrets["api_keys"]["PINECONE_API_KEY"]
aws_access_key_id = st.secrets["aws"]["aws_access_key_id"]
aws_secret_access_key = st.secrets["aws"]["aws_secret_access_key"]
aws_region = st.secrets["aws"]["aws_region"]

# Set environment variable for OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize boto3 S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

# Pull the retrieval QA chat prompt
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

# Initialize the ChatOpenAI instances with the API key
llm_inference = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=os.environ["OPENAI_API_KEY"])
llm_groundtruth = ChatOpenAI(model="gpt-4", openai_api_key=os.environ["OPENAI_API_KEY"])  # Use a different LLM for ground truth

# Initialize the retriever using PineconeVectorStore
model_name = "voyage-large-2"
embedding_function = VoyageAIEmbeddings(
    model=model_name,
    voyage_api_key= VOYAGE_AI_API_KEY
)
vector_store = PineconeVectorStore.from_existing_index(
    embedding=embedding_function,
    index_name="test"
)
retriever = vector_store.as_retriever()

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever_inference = create_history_aware_retriever(
    llm_inference, retriever, contextualize_q_prompt
)

history_aware_retriever_groundtruth = create_history_aware_retriever(
    llm_groundtruth, retriever, contextualize_q_prompt
)

question_answer_chain_inference = create_stuff_documents_chain(llm_inference, retrieval_qa_chat_prompt)
question_answer_chain_groundtruth = create_stuff_documents_chain(llm_groundtruth, retrieval_qa_chat_prompt)

rag_retreival_chain_inference = create_retrieval_chain(history_aware_retriever_inference, question_answer_chain_inference)
rag_retreival_chain_groundtruth = create_retrieval_chain(history_aware_retriever_groundtruth, question_answer_chain_groundtruth)

# Initialize memory
memory = ConversationBufferMemory()

# Store for chat history
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain_inference = RunnableWithMessageHistory(
    rag_retreival_chain_inference,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

conversational_rag_chain_groundtruth = RunnableWithMessageHistory(
    rag_retreival_chain_groundtruth,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

def retrieve_and_format_response(user_input, retriever, llm_chain):
    response = llm_chain.invoke({"input": user_input}, config={"configurable": {"session_id": "test"}})
    return response

# Additional Functions for Evaluation Metrics from Code 1

def bleu_score(reference, hypothesis):
    reference_tokens = [nltk.word_tokenize(reference)]
    hypothesis_tokens = nltk.word_tokenize(hypothesis)
    smoothing_function = SmoothingFunction().method1
    return sentence_bleu(reference_tokens, hypothesis_tokens, smoothing_function=smoothing_function)

def edit_distance(reference, hypothesis):
    m = len(reference) + 1
    n = len(hypothesis) + 1

    # Create a matrix to store the distances
    dp = np.zeros((m, n), dtype=int)

    # Initialize the first row and column
    for i in range(m):
        dp[i][0] = i
    for j in range(n):
        dp[0][j] = j

    # Compute the edit distance
    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if reference[i-1] == hypothesis[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,        # Deletion
                           dp[i][j-1] + 1,        # Insertion
                           dp[i-1][j-1] + cost)   # Substitution

    return dp[m-1][n-1]

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat messages from history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
user_input = st.chat_input("You: ")

if user_input:
    # Add user message to chat history
    st.session_state["messages"].append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate and display bot response
    with st.spinner("Thinking..."):
        inf_response = retrieve_and_format_response(user_input, retriever, conversational_rag_chain_inference)["answer"]
        gt_response = retrieve_and_format_response(user_input, retriever, conversational_rag_chain_groundtruth)["answer"]
    
    st.session_state["messages"].append({"role": "assistant", "content": inf_response})
    
    with st.chat_message("assistant"):
        st.markdown(inf_response)
    
    # Retrieve documents and prepare the content for download
    docs = retriever.get_relevant_documents(user_input)  # Use retriever to get documents
    retrieved_docs_content = save_retrieved_docs_to_file(docs)
    
    # Dynamically replace URL placeholders with pre-signed URLs
    retrieved_docs_content = replace_placeholders_with_urls(retrieved_docs_content, s3_client)
    
    # Display a download button for the retrieved documents
    st.download_button(
        label="More Info",
        data=retrieved_docs_content,
        file_name="retrieved_documents.json",
        mime="application/json"
    )

    # Evaluation
    eval_data = {
        "question": [user_input],
        "contexts": [[" ".join([d.page_content for d in docs])]],  # Combine retrieved documents into a single string and put into a list
        "answer": [inf_response],
        "ground_truth": [gt_response]
    }

    dataset_eval = Dataset.from_pandas(pd.DataFrame(eval_data))

    st.write("#### Debug: Dataset for Evaluation")
    st.write(eval_data)

    try:
        result = evaluate(
            dataset_eval,
            metrics=[
                context_relevancy,
                faithfulness,
                answer_relevancy,
                context_recall,
                harmfulness,
                answer_correctness
            ],
            raise_exceptions=False  # Don't raise exceptions
        )
        eval_df = result.to_pandas()

        st.write("### Evaluation Metrics")
        st.write("BLEU score:", round(bleu_score(gt_response, inf_response), 6))
        st.write("Edit distance:", edit_distance(gt_response, inf_response))
        st.write("Context relevancy:", round(eval_df.context_relevancy.loc[0], 6))
        st.write("Faithfulness:", eval_df.faithfulness.loc[0])
        st.write("Answer relevancy:", round(eval_df.answer_relevancy.loc[0], 6))
        st.write("Answer correctness:", eval_df.answer_correctness.loc[0])
        st.write("Context recall:", round(eval_df.context_recall.loc[0], 6))
        st.write("Harmfulness:", round(eval_df.harmfulness.loc[0], 6))
    except Exception as e:
        st.error(f"Error in evaluation: {str(e)}")
        st.error(traceback.format_exc())  # Print the full traceback for debugging

# Add an "End Conversation" button
if st.button("End Conversation"):
    # Save chat history to a file and upload to S3
    session_id = str(uuid.uuid4())
    chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state["messages"]])
    local_filename = f"chat_history_{session_id}.txt"
    with open(local_filename, 'w') as file:
        file.write(chat_history)
    chat_history_key = f"raw-data/chat_history_{session_id}.txt"
    s3_client.upload_file(local_filename, "chat-history-process", chat_history_key)
    st.success(f"Chat history saved and uploaded to S3 as '{chat_history_key}'")
    # Clear chat history from session state
    st.session_state["messages"] = []
