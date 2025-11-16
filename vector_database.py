from AI_Lawyer.utils.logging_setup import *
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

pdfs_directory = 'pdfs/' 
def upload_pdf(file):
    try:
        with open( pdfs_directory + file.name, "wb") as f:
         f.write(file.getbuffer())
         logger.info(f"file '{file.name}' uploaded successfully to '{pdfs_directory}'")

    except Exception as e:
       logger.error(f"error while uploading file '{file.name}'")

file_path = "universal_declaration_of_human_rights.pdf"


def load_pdf(file_path):
   try:
      loader = PDFPlumberLoader(file_path)
      documents = loader.load()   
      logger.info(f"The file in path '{file_path}' is succefully loaded")
      return documents
   
   except Exception as e:
      logger.error(f"error while loading the file  to path '{file_path}' ")
      
documents = load_pdf(file_path)

      
#step 2: Create Chunks 
def create_chunks(documents):
   tex_spillter = RecursiveCharacterTextSplitter(
      chunk_size = 1000,
      chunk_overlap = 200,
      add_start_index = True
   )

   text_chunks = tex_spillter.split_documents(documents)

   return text_chunks
    
text_chunks = create_chunks(documents)

print("Chunks count: " ,len(text_chunks))



#Step 3: Setup Embeddings Model (Use DeepSeek R1 with Ollama)

ollama_model_name = "deepseek-r1:1.5b"
def get_embedding_model(ollama_model_name):
   embeddings = OllamaEmbeddings(model=ollama_model_name)
   return embeddings


#Step 3: Setup Embeddings Model (Use DeepSeek R1 with Ollama)

FAISS_DB_PATH = "vectorstore/db_faiss"
faiss_db=FAISS.from_documents(text_chunks, get_embedding_model(ollama_model_name))
faiss_db.save_local(FAISS_DB_PATH)



def create_vector_store(db_faiss_path, text_chunks, ollama_model_name):
   faiss_db = FAISS.from_documents(text_chunks, get_embedding_model(ollama_model_name))
   faiss_db.save_local(db_faiss_path)
   return faiss_db




    




   
   
   
   
