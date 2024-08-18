import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
import os

# Load environment variables
load_dotenv()

# Configure the Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the Gemini model
gemini_model = genai.GenerativeModel('gemini-pro')

# Function to extract text from uploaded PDFs with page numbers
def get_pdf_text(pdf_docs):
    text_with_pages = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages, 1):
            text = page.extract_text()
            text_with_pages.append((text, page_num))
    return text_with_pages

# Function to split text into manageable chunks
def get_text_chunks(text_with_pages):
    chunks_with_pages = []
    for text, page_num in text_with_pages:
        chunks_with_pages.append((text, page_num))
    return chunks_with_pages

# Function to store vector embeddings
def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        texts = [chunk[0] for chunk in text_chunks]
        metadatas = [{"page": chunk[1]} for chunk in text_chunks]
        vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")

# Function to load the question-answering chain
def get_conversation_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the context, just say, "answer is not available in the context."
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function for the Q&A process
def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversation_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        
        # Extract page numbers from the relevant documents
        page_numbers = set(doc.metadata['page'] for doc in docs)
        page_info = f"This answer is derived from page(s): {', '.join(map(str, sorted(page_numbers)))}"
        
        st.write("Reply: ", response["output_text"])
        st.write(page_info)
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

# Recursive Summarization Function
def recursive_summarize(text_with_pages):
    try:
        # Summarize each page
        page_summaries = []
        for text, page_num in text_with_pages:
            summary_prompt = f"Summarize the following text from page {page_num}: \n\n{text[:4000]}"
            response = gemini_model.generate_content(summary_prompt)
            page_summaries.append(f"Page {page_num} Summary: {response.text}")

        # Combine page summaries
        combined_summaries = "\n\n".join(page_summaries)

        # Create final summary
        final_summary_prompt = f"Provide a comprehensive summary of the entire document based on these page summaries:\n\n{combined_summaries}"
        final_response = gemini_model.generate_content(final_summary_prompt)
        return final_response.text
    except Exception as e:
        st.error(f"An error occurred during summarization: {str(e)}")
        return None

# Main function for Streamlit app
def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with PDF using Gemini")

    # Sidebar for PDF upload
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        process_button = st.button("Submit & Process")

    if process_button:
        if pdf_docs:
            with st.spinner("Processing..."):
                text_with_pages = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(text_with_pages)
                get_vector_store(text_chunks)
                st.session_state['processed'] = True
                st.session_state['text_with_pages'] = text_with_pages
                st.success("Processing complete!")
        else:
            st.error("Please upload a PDF file.")

    option = st.selectbox("Choose an option", ["Summarize the PDF", "Ask a Question"])

    if option == "Summarize the PDF":
        if st.button("Summarize"):
            if 'processed' in st.session_state and st.session_state['processed']:
                with st.spinner("Summarizing the content..."):
                    summary = recursive_summarize(st.session_state['text_with_pages'])
                    if summary:
                        st.subheader("Summary of the PDF")
                        st.write(summary)
            else:
                st.error("Please upload and process a PDF file first.")

    elif option == "Ask a Question":
        user_question = st.text_input("Ask a Question:")
        if user_question:
            if 'processed' in st.session_state and st.session_state['processed']:
                with st.spinner("Fetching the answer..."):
                    user_input(user_question)
            else:
                st.error("Please upload and process a PDF file first.")

if __name__ == "__main__":
    main()
