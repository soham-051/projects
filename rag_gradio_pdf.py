import gradio as gr
import fitz  # PyMuPDF
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import traceback

# Step 1: Extract text from PDF
def parse_pdf(file_path):
    print(f"[INFO] Parsing PDF: {file_path}")
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    print(f"[INFO] Extracted {len(full_text)} characters from PDF.")
    return full_text

# Step 2: Split text into smaller chunks
def split_text(text):
    print("[INFO] Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.create_documents([text])
    print(f"[INFO] Split into {len(chunks)} chunks.")
    return chunks

# Step 3: Build vector database
def build_vectorstore(docs):
    print("[INFO] Creating vector store...")
    embeddings = OllamaEmbeddings(model="llama3.1")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# Global variable to store QA chain
qa_chain = None

# Load PDF and initialize RAG pipeline
def load_pdf_to_qa_chain(file_obj):
    global qa_chain
    try:
        file_path = file_obj.name
        text = parse_pdf(file_path)
        docs = split_text(text)
        vectorstore = build_vectorstore(docs)
        retriever = vectorstore.as_retriever()
        llm = Ollama(model="llama3.1")
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        return "✅ PDF loaded and indexed."
    except Exception as e:
        error_message = f"❌ Failed to load PDF: {str(e)}"
        print(error_message)
        traceback.print_exc()
        return error_message

# Handle user question
def answer_question(question):
    try:
        if qa_chain is None:
            return "⚠️ Please upload and load a PDF first."
        print(f"[INFO] Answering question: {question}")
        answer = qa_chain.run(question)
        return answer
    except Exception as e:
        error_message = f"❌ Error during question answering: {str(e)}"
        print(error_message)
        traceback.print_exc()
        return error_message

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📄 Chat with your PDF using Ollama + LLaMA 3.1 + RAG")
    
    file_input = gr.File(label="Upload PDF", type="filepath")
    upload_btn = gr.Button("Load PDF into RAG")
    
    question_box = gr.Textbox(label="Ask a question about the PDF", placeholder="E.g., What is the architecture of the system?")
    answer_box = gr.Textbox(label="Answer", interactive=False)
    
    upload_btn.click(load_pdf_to_qa_chain, inputs=file_input, outputs=answer_box)
    question_box.submit(answer_question, inputs=question_box, outputs=answer_box)

demo.launch(share=True)
