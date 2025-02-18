import sys
import os
import chainlit as cl
import lancedb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from lancedb.pydantic import LanceModel, Vector
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from lancedb.pydantic import LanceModel
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
import nest_asyncio
from helperr import calculate_similarity, load_key


transformer_model = SentenceTransformer('BAAI/bge-small-en-v1.5')


class DocumentEmbedding(LanceModel):
    document_id: str
    chunk_text: str
    vector: Vector(transformer_model.get_sentence_embedding_dimension())


def create_fill_table(db):
    import os
    print("Applying nest_asyncio...")
    nest_asyncio.apply()

    print("Loading API Key...")
    API_KEY = load_key('API_KEY')

    print("Initializing LlamaParse...")
    llama_parse = LlamaParse(api_key=API_KEY)

    print("Setting Up File Extractor...")
    file_extractor = {".pdf": llama_parse}

    pdf_path = os.path.join(os.getcwd(), "data",
                            "The New Complete Book of Foos.pdf")
    if not os.path.exists(pdf_path):
        print("PDF file not found at:", pdf_path)
        raise FileNotFoundError(f"File not found: {pdf_path}")
    else:
        print("PDF file found, attempting to load...")

    documents = SimpleDirectoryReader(
        input_files=[pdf_path], file_extractor=file_extractor).load_data()

    table = db.create_table("document_embeddings", schema=DocumentEmbedding)
    chunk_size = 2000
    chunk_overlap = 200
    for doc in documents:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(doc.text)

        embeddings = [transformer_model.encode(chunk) for chunk in chunks]

        data = [
            DocumentEmbedding(document_id=doc.id_,
                              chunk_text=chunk, vector=embedding.tolist())
            for embedding, chunk in zip(embeddings, chunks)
        ]
        table.add(data)
    return table


def answering_general_questions(user_input):
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from chatbot_main import Internet_Result, Local_Result
    db = lancedb.connect("./data/my_lancedb")
    try:
        table = db.open_table("document_embeddings")
    except:
        table = create_fill_table(db)
    query_vector = transformer_model.encode(user_input).tolist()

    results = table.search(query_vector).limit(
        2).to_pydantic(DocumentEmbedding)

    document_vectors = [result.vector for result in results]
    similarities = calculate_similarity(query_vector, document_vectors)

    SIMILARITY_THRESHOLD = 0.5
    cl.user_session.set("current_state", None)
    if max(similarities) < SIMILARITY_THRESHOLD:
        print("Internet_Result")
        internet_results = Internet_Result(user_input)
        return internet_results
    else:
        print("Local_Result")
        local_results = Local_Result(user_input, table, transformer_model)
        return local_results
