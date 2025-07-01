
# import os
# import re
# from typing import List
# from pathlib import Path

# # from dotenv import load_dotenv
# from pypdf import PdfReader
# from langchain.docstore.document import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# # from langchain_openai import OpenAIEmbeddings
# from langchain.embeddings import OpenAIEmbeddings  # ✅
# from dotenv import load_dotenv
# import os
# import openai

# load_dotenv()


# # Dynamically resolve base project directory
# BASE_DIR = Path(__file__).resolve().parent.parent.parent

# # Function to parse PDF and extract text content
# def parse_pdf(file_path: str) -> List[str]:
#     with open(file_path, 'rb') as file:
#         pdf = PdfReader(file)
#         output = []
#         for page in pdf.pages:
#             text = page.extract_text()
#             if not text:
#                 continue
#             text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)  # Merge hyphenated words
#             text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())  # Fix newlines
#             text = re.sub(r"\n\s*\n", "\n\n", text)  # Remove multiple newlines
#             output.append(text)
#         return output

# # Function to convert text content into documents
# def text_to_docs(text: List[str], source: str) -> List[Document]:
#     page_docs = [Document(page_content=page) for page in text]
#     for i, doc in enumerate(page_docs):
#         doc.metadata["page"] = i + 1
#         doc.metadata["source_pdf"] = source

#     doc_chunks = []
#     for doc in page_docs:
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=2000,
#             separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
#             chunk_overlap=0,
#         )
#         chunks = text_splitter.split_text(doc.page_content)
#         for i, chunk in enumerate(chunks):
#             chunk_doc = Document(
#                 page_content=chunk,
#                 metadata={
#                     "page": doc.metadata["page"],
#                     "chunk": i,
#                     "source_pdf": doc.metadata["source_pdf"],
#                     "source": f"{doc.metadata['source_pdf']} (Page {doc.metadata['page']}-{i})"
#                 },
#             )
#             doc_chunks.append(chunk_doc)
#     return doc_chunks

# # Save processed PDF to FAISS index directory
# def save_pdf_to_faiss(file_path: str, api_key: str, faiss_folder: str):
#     file_path = Path(file_path)
#     faiss_folder = Path(faiss_folder)

#     file_name = file_path.name
#     faiss_path = faiss_folder / file_name.replace(".pdf", "")
    
#     print(f"\n📄 Processing: {file_name}")

#     try:
#         text = parse_pdf(str(file_path))
#         doc_chunks = text_to_docs(text, source=file_name)

#         if doc_chunks:
#             print(f"✅ Loaded {len(doc_chunks)} chunks from {file_name}")

#             # embeddings = OpenAIEmbeddings(openai_api_key=api_key,model = "gpt-3.5turbo")
#             # embeddings = OpenAIEmbeddings() 
#             embeddings = OpenAIEmbeddings(openai_api_key=api_key, model=os.getenv("OPENAI_API_MODEL")

# ,base_url=os.getenv("OPENAI_API_BASE_URL"))
#             index = FAISS.from_documents(doc_chunks, embeddings)

#             faiss_path.mkdir(parents=True, exist_ok=True)
#             index.save_local(str(faiss_path))

#             print(f"💾 Saved FAISS index for {file_name} to {faiss_path}\n")
#         else:
#             print(f"⚠️ No chunks generated for {file_name}")

#     except Exception as e:
#         import traceback
#         print(f"❌ Error processing {file_name}:\n{traceback.format_exc()}")

# # Main function to process all PDFs in a folder
# def process_all_pdfs(pdf_folder: str, faiss_folder: str, api_key: str):
#     pdf_folder = Path(pdf_folder)
#     faiss_folder = Path(faiss_folder)
#     faiss_folder.mkdir(parents=True, exist_ok=True)

#     for file_path in pdf_folder.glob("*.pdf"):
#         save_pdf_to_faiss(str(file_path), api_key, str(faiss_folder))

# # Run script
# if __name__ == "__main__":
#     PDF_FOLDER_PATH = BASE_DIR / "data" / "pdfs"
#     FAISS_FOLDER_PATH = BASE_DIR / "data" / "faiss_indexes"

#     PDF_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
#     FAISS_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
#     openai.api_key = "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"


#     # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#     # openai= "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"
   

#     process_all_pdfs(PDF_FOLDER_PATH, FAISS_FOLDER_PATH, openai.api_key)




# Emmu

import os
import re
from typing import List
from pathlib import Path

from pypdf import PdfReader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ FREE embeddings
from dotenv import load_dotenv

load_dotenv()

# Dynamically resolve base project directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Function to parse PDF and extract text content
def parse_pdf(file_path: str) -> List[str]:
    """Extract text content from PDF file"""
    print(f"📖 Reading PDF: {Path(file_path).name}")
    
    with open(file_path, 'rb') as file:
        pdf = PdfReader(file)
        output = []
        
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if not text:
                print(f"⚠️ No text found on page {page_num}")
                continue
                
            # Clean up text formatting
            text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)  # Merge hyphenated words
            text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())  # Fix newlines
            text = re.sub(r"\n\s*\n", "\n\n", text)  # Remove multiple newlines
            output.append(text)
            
        print(f"✅ Extracted text from {len(output)} pages")
        return output

# Function to convert text content into documents
def text_to_docs(text: List[str], source: str) -> List[Document]:
    """Convert extracted text into LangChain documents with metadata"""
    print(f"📝 Converting text to documents for: {source}")
    
    # Create page documents
    page_docs = [Document(page_content=page) for page in text]
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1
        doc.metadata["source_pdf"] = source

    # Split documents into chunks
    doc_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Reduced for better performance
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        chunk_overlap=200,  # Added overlap for better context
    )
    
    for doc in page_docs:
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            chunk_doc = Document(
                page_content=chunk,
                metadata={
                    "page": doc.metadata["page"],
                    "chunk": i,
                    "source_pdf": doc.metadata["source_pdf"],
                    "source": f"{doc.metadata['source_pdf']} (Page {doc.metadata['page']}-{i})"
                },
            )
            doc_chunks.append(chunk_doc)
    
    print(f"✅ Created {len(doc_chunks)} document chunks")
    return doc_chunks

# Initialize embeddings once (reuse for all PDFs)
def get_embeddings():
    """Get Hugging Face embeddings - FREE and runs locally"""
    print("🔄 Initializing Hugging Face embeddings...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Fast & lightweight
        model_kwargs={'device': 'cpu'},  # Change to 'cuda' if you have GPU
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print("✅ Embeddings initialized successfully")
    return embeddings

# Save processed PDF to FAISS index directory
def save_pdf_to_faiss(file_path: str, faiss_folder: str, embeddings):
    """Process single PDF and save to FAISS index"""
    file_path = Path(file_path)
    faiss_folder = Path(faiss_folder)

    file_name = file_path.name
    faiss_path = faiss_folder / file_name.replace(".pdf", "")
    
    print(f"\n{'='*60}")
    print(f"📄 Processing: {file_name}")
    print(f"💾 Output path: {faiss_path}")
    print(f"{'='*60}")

    try:
        # Extract text from PDF
        text = parse_pdf(str(file_path))
        
        if not text:
            print(f"⚠️ No text extracted from {file_name}")
            return False
            
        # Convert to documents
        doc_chunks = text_to_docs(text, source=file_name)

        if doc_chunks:
            print(f"🔄 Creating FAISS index with {len(doc_chunks)} chunks...")
            
            # Create FAISS index with FREE embeddings
            index = FAISS.from_documents(doc_chunks, embeddings)

            # Save index
            faiss_path.mkdir(parents=True, exist_ok=True)
            index.save_local(str(faiss_path))

            print(f"✅ FAISS index saved for {file_name}")
            print(f"📊 Index stats:")
            print(f"   - Total vectors: {index.index.ntotal}")
            print(f"   - Vector dimensions: {index.index.d}")
            print(f"   - Storage path: {faiss_path}")
            
            return True
        else:
            print(f"❌ No chunks generated for {file_name}")
            return False

    except Exception as e:
        import traceback
        print(f"❌ Error processing {file_name}:")
        print(f"Error: {str(e)}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return False

# Main function to process all PDFs in a folder
def process_all_pdfs(pdf_folder: str, faiss_folder: str):
    """Process all PDFs in the folder and create FAISS indexes"""
    pdf_folder = Path(pdf_folder)
    faiss_folder = Path(faiss_folder)
    
    # Create directories if they don't exist
    faiss_folder.mkdir(parents=True, exist_ok=True)
    
    # Check if PDF folder exists
    if not pdf_folder.exists():
        print(f"❌ PDF folder not found: {pdf_folder}")
        print("Please create the folder and add your PDF files.")
        return
    
    # Find all PDF files
    pdf_files = list(pdf_folder.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in: {pdf_folder}")
        return
    
    print(f"🚀 Found {len(pdf_files)} PDF files to process")
    print(f"📁 PDF folder: {pdf_folder}")
    print(f"💾 FAISS output folder: {faiss_folder}")
    
    # Initialize embeddings once (more efficient)
    embeddings = get_embeddings()
    
    # Process each PDF
    successful = 0
    failed = 0
    
    for i, file_path in enumerate(pdf_files, 1):
        print(f"\n🔄 Processing file {i}/{len(pdf_files)}: {file_path.name}")
        
        success = save_pdf_to_faiss(str(file_path), str(faiss_folder), embeddings)
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {successful} files")
    print(f"❌ Failed to process: {failed} files")
    print(f"📁 FAISS indexes saved to: {faiss_folder}")
    
    if successful > 0:
        print(f"\n🎉 Success! You can now use these FAISS indexes for RAG queries.")
        print(f"💡 Next step: Update your query code to use the same HuggingFace embeddings.")

# Test function to verify a specific index
def test_faiss_index(faiss_path: str, test_query: str = "What is this document about?"):
    """Test a FAISS index to make sure it works"""
    faiss_path = Path(faiss_path)
    
    if not faiss_path.exists():
        print(f"❌ FAISS index not found: {faiss_path}")
        return
    
    try:
        print(f"🧪 Testing FAISS index: {faiss_path}")
        
        # Use same embeddings as used for creation
        embeddings = get_embeddings()
        
        # Load the index
        vectorstore = FAISS.load_local(str(faiss_path), embeddings, allow_dangerous_deserialization=True)
        
        print(f"✅ Index loaded successfully!")
        print(f"📊 Total vectors: {vectorstore.index.ntotal}")
        print(f"📐 Vector dimensions: {vectorstore.index.d}")
        
        # Test similarity search
        docs = vectorstore.similarity_search(test_query, k=2)
        
        print(f"\n🔍 Test query: '{test_query}'")
        print(f"📄 Found {len(docs)} relevant documents:")
        
        for i, doc in enumerate(docs, 1):
            print(f"\nResult {i}:")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Content preview: {doc.page_content[:200]}...")
            
    except Exception as e:
        print(f"❌ Error testing index: {e}")

# Run script
if __name__ == "__main__":
    # Define paths
    PDF_FOLDER_PATH = BASE_DIR / "data" / "pdfs"
    FAISS_FOLDER_PATH = BASE_DIR / "data" / "faiss_indexes"

    # Create directories
    PDF_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
    FAISS_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
    
    print("🚀 PDF to FAISS Converter (Using FREE Hugging Face Embeddings)")
    print(f"📁 PDF folder: {PDF_FOLDER_PATH}")
    print(f"💾 FAISS folder: {FAISS_FOLDER_PATH}")
    
    # Process all PDFs
    process_all_pdfs(PDF_FOLDER_PATH, FAISS_FOLDER_PATH)
    
    # Optional: Test one of the created indexes
    # Uncomment the lines below to test
    test_index_path = FAISS_FOLDER_PATH / "your_pdf_name_here"
    test_faiss_index(test_index_path)