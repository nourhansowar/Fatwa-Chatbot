from FAISS import *
import openai
import streamlit as st
from tokens import OPENAI_API, MODEL_NAME, INDEX_PATH, METADATA_PATH
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set the OpenAI API token and model name
client = openai.OpenAI(api_key=OPENAI_API)

# Cache the FAISS model to avoid reloading it multiple times
@st.cache_resource()
def load_model():
    return FAISSEmbedding()

# Load the FAISS model
Faiss_wrapper = load_model()

# Function to stream chat responses
def stream_chat_response(message, chat_history, system_msg_content, model_name, temperature, max_history_length):
    # Create system message with initial content
    system_msg = [{"role": "system", "content": system_msg_content}]
    # Append user message to chat history
    chat_history.append({"role": "user", "content": message})
    # Ensure chat history does not exceed max length
    if len(chat_history) > max_history_length:
        chat_history = chat_history[-max_history_length:]
    messages = system_msg + chat_history

    # Create the chat completion stream
    stream = client.chat.completions.create(
        messages=messages,
        model=model_name,
        temperature=temperature,
        stream=True
    )

    # Yield each chunk of the response
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Function to clear the chat history
def clear_chat():
    st.session_state.chat_history = []

# Function to format the context returned by the FAISS search
def format_context(context):
    formatted_context = ""
    for i, doc in enumerate(context):
        formatted_context += f"**Context {i+1}:** {doc.page_content}\n\n"
    return formatted_context

def main():
    st.title("💬 Chat with AI - RAG Model and Vector DB")

    # Sidebar controls
    model_name = st.sidebar.selectbox("Choose the Model", ["text-embedding-3-large", "gpt-3.5-turbo", "text-embedding-ada-002"], index=1)

    temperature = st.sidebar.slider("Set Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    max_history_length = int(st.sidebar.number_input("Max History Length", min_value=1, max_value=10, value=3))
    system_msg = st.sidebar.text_area("System Message (Persona)", value="", height=100)

    # Initialize the vector database in the session state if not already present
    if 'vec_db' not in st.session_state:
        st.session_state.vec_db = None

    # Clear chat button in the sidebar
    if st.sidebar.button("Clear Chat"):
        clear_chat()

    # Initialize chat history in the session state if not already present
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display existing chat history
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    # Input field for user message
    user_input = st.chat_input("Enter your message:", key="user_input")
    
    if user_input:
        st.chat_message("user").write(user_input)
        with st.spinner("Thinking..."):
            accumulated_response = ""
            placeholder = st.chat_message("AI").empty()

            # Append extra context to the system message
            system_msg += "\nUse the following extra context :\n{context}" 
            context = Faiss_wrapper.search(query=user_input)
            ################################
            ## SUMARIZATION PART HERE (CONTEXT: LIST OF most_similar_entries)


            ################################

            if context != None:                           
                system_msg = system_msg.format(context=context)
                st.session_state.last_context = context
            else: 
                system_msg = system_msg.format(context="No context found")
                st.session_state.last_context = None

            # Get the response in chunks and display it
            for response_chunk in stream_chat_response(user_input, 
                                                       st.session_state.chat_history, 
                                                       system_msg, 
                                                       model_name, 
                                                       temperature, 
                                                       max_history_length):
                accumulated_response += response_chunk
                placeholder.markdown(accumulated_response)
            st.session_state.chat_history.append({"role": "assistant", "content": accumulated_response})

            # Display the last query relevant context in the sidebar
            if 'last_context' in st.session_state and st.session_state.last_context != None:
                formatted_context = format_context(st.session_state.last_context)
                st.sidebar.text_area("Last query relevant context:", value=formatted_context, height=300)
                st.session_state.last_context = None

if __name__ == "__main__":
    main()
