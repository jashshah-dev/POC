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

def main():
    st.set_page_config(page_title="Chat PDF", page_icon="📚", layout="wide")
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5em;
        color: #4A4A4A;
        text-align: center;
        margin-bottom: 30px;
    }
    .sub-header {
        font-size: 1.5em;
        color: #6A6A6A;
        margin-bottom: 20px;
    }
    .info-box {
        background-color: #F0F2F6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .step {
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='main-header'>📚 PDF Genius: Your Smart Document Assistant</h1>", unsafe_allow_html=True)

    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>How PDF Genius Works:</h2>", unsafe_allow_html=True)
    st.markdown("""
    <ol>
    <li class='step'><strong>Upload:</strong> Start by uploading one or more PDF files.</li>
    <li class='step'><strong>Process:</strong> Click 'Submit & Process' to analyze your documents.</li>
    <li class='step'><strong>Interact:</strong> Choose to summarize the entire document or ask specific questions.</li>
    </ol>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("<h2 class='sub-header'>📤 Upload & Process</h2>", unsafe_allow_html=True)
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True, type="pdf")
        
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing your documents... This may take a moment."):
                    text_with_pages = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(text_with_pages)
                    get_vector_store(text_chunks)
                    st.session_state['processed'] = True
                    st.session_state['text_with_pages'] = text_with_pages
                st.success("✅ Processing complete! You can now summarize or ask questions.")
            else:
                st.error("Please upload at least one PDF file.")
        
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown("<h3>Why Submit & Process?</h3>", unsafe_allow_html=True)
        st.markdown("""
        This step is crucial as it:
        - Extracts text from your PDFs
        - Analyzes the content
        - Prepares the data for quick retrieval
        
        Without this step, the app can't understand or answer questions about your documents.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<h2 class='sub-header'>🧠 Interact with Your Documents</h2>", unsafe_allow_html=True)
        
        option = st.selectbox("Choose an option:", ["Summarize the PDF", "Ask a Question"])

        if option == "Summarize the PDF":
            if st.button("Generate Summary"):
                if 'processed' in st.session_state and st.session_state['processed']:
                    with st.spinner("Crafting a comprehensive summary..."):
                        summary = recursive_summarize(st.session_state['text_with_pages'])
                        if summary:
                            st.markdown("<h3>📝 Document Summary:</h3>", unsafe_allow_html=True)
                            st.write(summary)
                else:
                    st.warning("Please upload and process your PDF files first.")

        elif option == "Ask a Question":
            user_question = st.text_input("What would you like to know about your document?")
            if user_question:
                if 'processed' in st.session_state and st.session_state['processed']:
                    with st.spinner("Searching for the best answer..."):
                        user_input(user_question)
                else:
                    st.warning("Please upload and process your PDF files first.")

    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    st.markdown("<h3>💡 How PDF Genius Helps You:</h3>", unsafe_allow_html=True)
    st.markdown("""
    - **Time-Saving:** Quickly summarize lengthy documents.
    - **Insightful:** Extract key information without reading the entire text.
    - **Efficient:** Ask specific questions and get precise answers.
    - **Versatile:** Works with multiple PDFs, ideal for research or document analysis.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
