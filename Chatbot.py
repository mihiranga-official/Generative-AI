import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain  # Adjust if LLMChain is available
from langchain.chat_models import ChatOpenAI

# Initialize OpenAI API key
OPENAI_API_KEY = "sk-None-D8O6HrBm8hQBJiW43TEWT3BlbkFJi3yVyFg88ndaCYTRLCkv"

# Function to get embeddings using LangChain's OpenAIEmbeddings
def get_embeddings(text_chunks):
    embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return [embeddings_model.embed(chunk) for chunk in text_chunks]

# Upload PDF File
st.header("Chat Bot by Janith")
with st.sidebar:
    st.title("Your Document")
    file = st.file_uploader("Choose a file", type="pdf")

    # Extract the text
    if file is not None:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n"],
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.write(chunks)

        # Generate embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # Create a vector store using FAISS
        try:
            vector_store = FAISS.from_texts(chunks, embeddings)
        except Exception as e:
            st.error(f"Error creating vector store: {e}")

        # Get user question
        user_question = st.text_input("Type your question here")

        # Perform similarity search
        if user_question:
            try:
                match = vector_store.similarity_search(user_question)
                # st.write(match)  # Uncomment to display the match
            except Exception as e:
                st.error(f"Error during similarity search: {e}")

            # Define the LLM
            llm = ChatOpenAI(
                openai_api_key=OPENAI_API_KEY,
                temperature=0,
                max_tokens=1000,
                model_name="gpt-3.5-turbo"
            )

            # Output results
            try:
                # Use LLMChain or appropriate class
                chain = LLMChain(llm=llm)
                response = chain.run(input_documents=match, question=user_question)
                st.write(response)
            except Exception as e:
                st.error(f"Error generating response: {e}")
