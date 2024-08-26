import streamlit as st
from audio_recorder_streamlit import audio_recorder
from openai import OpenAI
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from pathlib import Path


# Load environment variables
#load_dotenv()
#os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
#os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
openai_api_key = st.secrets['OPENAI_API_KEY']

# Initialize OpenAI client
client = OpenAI()

# Define self-reference queries
self_reference_queries = [
    "who are you", 
    "what is your name", 
    "who is this", 
    "what are you"
]

# Initialize models and data using the new pipeline
@st.cache_resource
def load_models():
    groq_api_key = st.secrets['GROQ_API_KEY']
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-70b-8192")

    # Load and split documents
    loader = PyPDFDirectoryLoader('./knowledge_base')
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    # Contextualize question system prompt
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

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Answer question
    system_prompt = (
    "You are a helpful assistant designed to answer questions for children aged 7 to 14. "
    "Always start by trying to find the answer in the provided information. "
    "If the answer isn't found there, you can use your knowledge to provide a response, "
    "but make sure it's safe, fun, and easy to understand for kids. "
    "Please provide the answer directly without any introductory phrases."
    "\n\n"
    "{context}"
    "\n\n"
    "Question: {input}\n"
    "Answer directly: "
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    return conversational_rag_chain

# Load the conversational retrieval-aware chain
conversational_rag_chain = load_models()

# Function to transcribe audio using OpenAI Whisper API
def transcribe_audio(audio_bytes):
    audio_path = "audio_file1.mp3"
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)
    
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )
    
    return transcription.text

# Function to convert text to speech using OpenAI TTS API
def text_to_speech(text):
    speech_file_path = "response.mp3"
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    
    response.stream_to_file(speech_file_path)
    
    with open(speech_file_path, "rb") as f:
        audio_bytes = f.read()
    
    return audio_bytes

# Streamlit App Title and Description
st.title("🧑‍💻 Ashal 🌟✨🌍 Assistant 🤖")

"""
Hi🤖 just click on the voice recorder and let me know how I can help you today?
"""

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    st.session_state["initialized"] = False  # Add an initialized flag

# Force a rerun if not initialized
if not st.session_state["initialized"]:
    st.session_state["initialized"] = True
    st.experimental_rerun()

# Capture user input through audio recorder
audio_bytes = audio_recorder(key="recorder")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            st.audio(message["audio"], format="audio/wav")
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            st.audio(message["audio"], format="audio/mp3")

if audio_bytes:
    with st.spinner('Processing...'):
        # Transcribe the audio file to text using OpenAI Whisper API
        transcribed_text = transcribe_audio(audio_bytes)
        st.success("Audio file successfully transcribed!")

        # Check if the transcribed text is a self-reference query
        if any(query in transcribed_text.lower() for query in self_reference_queries):
            answer = "My name is Ashal, and I'm helpful assistant designed to answer questions for children aged 7 to 14."
        else:
            # Use the transcribed text as the query for the conversational retrieval-aware chain
            result = conversational_rag_chain.invoke(
                {"input": transcribed_text},
                config={
                    "configurable": {"session_id": "abc123"}
                }
            )
            answer = result['answer']

        # Convert the assistant's response to speech using OpenAI TTS API
        response_audio_bytes = text_to_speech(answer)

        # Append assistant's response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer, "audio": response_audio_bytes})

        # Display the transcribed text and the assistant's response
        with st.chat_message("user"):
            st.audio(audio_bytes, format="audio/wav")

        with st.chat_message("assistant"):
            st.audio(response_audio_bytes, format="audio/mp3")

# Option to clear chat history
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.experimental_rerun()
