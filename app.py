import streamlit as st
import os
import logging
from typing import List
import time

# Import the LangGraphChatbot from the main module
from vidhijan_agent import LangGraphChatbot, Config

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Page configuration
st.set_page_config(
    page_title="Vidhijan",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if "chatbot" not in st.session_state:
        with st.spinner("Initializing chatbot..."):
            st.session_state.chatbot = LangGraphChatbot()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def main():
    """Main function to run the Streamlit app"""
    # Initialize session state
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.title("⚖️ Vidhijan Settings")
        st.markdown("---")

        st.subheader("Model Settings")
        groq_model = st.selectbox(
            "Groq Model",
            ["llama-3.1-8b-instant", "mixtral-8x7b-32768", "llama-3.1-70b"],
            index=0,
        )

        ollama_model = st.selectbox(
            "Embeddings Model",
            ["all-minilm:33m", "all-minilm", "nomic-embed-text"],
            index=0,
        )

        if st.button("Apply Settings"):
            Config.GROQ_MODEL = groq_model
            Config.OLLAMA_MODEL = ollama_model
            st.session_state.chatbot = LangGraphChatbot()
            st.success("Settings applied!")

        st.markdown("---")
        st.markdown(
            "This assistant provides general legal information. It should not be considered as legal advice."
        )

        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.experimental_rerun()

    # Main content
    st.title("Vidhijan ")
    st.markdown("Ask questions about commercial laws and regulations.")

    # Display chat history using Streamlit's native chat elements
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant", avatar="⚖️"):
                st.write(message["content"])

    # Input area
    user_input = st.chat_input("Ask about commercial laws...")

    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Display the user message (since we're not using st.experimental_rerun())
        with st.chat_message("user"):
            st.write(user_input)

        # Display thinking indicator
        with st.chat_message("assistant", avatar="⚖️"):
            with st.status("Processing your question..."):
                # Process the query
                response = st.session_state.chatbot.process_query(user_input)

        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Display the assistant's response
        with st.chat_message("assistant", avatar="⚖️"):
            st.write(response)


if __name__ == "__main__":
    main()
