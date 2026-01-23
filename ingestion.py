import glob
import re
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

embedding_model = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={"device": "cpu", "trust_remote_code": True}
)

vector_store = Chroma(
    embedding_function=embedding_model,
    persist_directory="./chroma_db"
)

def cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two vectors.

    Args:
    vec1 (np.ndarray): First vector.
    vec2 (np.ndarray): Second vector.

    Returns:
    float: Cosine similarity.
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def compute_breakpoints(similarities, method="percentile", threshold=90):
    """
    Computes chunking breakpoints based on similarity drops.

    Args:
    similarities (List[float]): List of similarity scores between sentences.
    method (str): 'percentile', 'standard_deviation', or 'interquartile'.
    threshold (float): Threshold value (percentile for 'percentile', std devs for 'standard_deviation').

    Returns:
    List[int]: Indices where chunk splits should occur.
    """
    if not similarities:
        return []
    
    # Determine the threshold value based on the selected method
    if method == "percentile":
        # Calculate the Xth percentile of the similarity scores
        threshold_value = np.percentile(similarities, threshold)
    elif method == "standard_deviation":
        # Calculate the mean and standard deviation of the similarity scores
        mean = np.mean(similarities)
        std_dev = np.std(similarities)
        # Set the threshold value to mean minus X standard deviations
        threshold_value = mean - (threshold * std_dev)
    elif method == "interquartile":
        # Calculate the first and third quartiles (Q1 and Q3)
        q1, q3 = np.percentile(similarities, [25, 75])
        # Set the threshold value using the IQR rule for outliers
        threshold_value = q1 - 1.5 * (q3 - q1)
    else:
        # Raise an error if an invalid method is provided
        raise ValueError("Invalid method. Choose 'percentile', 'standard_deviation', or 'interquartile'.")

    # Identify indices where similarity drops below the threshold value
    return [i for i, sim in enumerate(similarities) if sim < threshold_value]

def split_into_chunks(sentences, breakpoints):
    """
    Splits sentences into semantic chunks.

    Args:
    sentences (List[str]): List of sentences.
    breakpoints (List[int]): Indices where chunking should occur.

    Returns:
    List[str]: List of text chunks.
    """
    chunks = [] 
    start = 0  
    # Iterate through each breakpoint to create chunks
    for bp in breakpoints:
        chunks.append(". ".join(sentences[start:bp + 1]) + ".")
        start = bp + 1  

    chunks.append(". ".join(sentences[start:]))
    return chunks  # Return the list of chunks


def semantic_split_text(text, method="percentile", threshold=90):
    """
    Performs semantic chunking on text by computing sentence embeddings
    and splitting at points where semantic similarity drops.
    
    Args:
    text (str): The text to split.
    method (str): The method for computing breakpoints ('percentile', 'standard_deviation', 'interquartile').
    threshold (float): The threshold value for the selected method.
    
    Returns:
    List[str]: List of semantically coherent chunks.
    """
    # Split text into sentences using simple regex
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= 1:
        return sentences
    
    # Get embeddings for all sentences
    sentence_embeddings = embedding_model.embed_documents(sentences)
    sentence_embeddings = np.array(sentence_embeddings)
    
    # Compute cosine similarity between consecutive sentences
    similarities = [
        cosine_similarity(sentence_embeddings[i], sentence_embeddings[i + 1]) 
        for i in range(len(sentence_embeddings) - 1)
    ]
    
    # Compute breakpoints based on similarity drops
    breakpoints = compute_breakpoints(similarities, method=method, threshold=threshold)
    
    # Split text into chunks using breakpoints
    chunks = split_into_chunks(sentences, breakpoints)
    
    return chunks


def load_docs(file_path):
    """
    Loads documents from a .docx file and applies semantic chunking.
    
    Args:
    file_path (str): Path to the .docx file.
    
    Returns:
    Chroma: Vector store with added documents.
    """
    if file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
        docs = loader.load()
        
        # Apply semantic chunking to each document
        semantic_chunks = []
        for doc in docs:
            chunks = semantic_split_text(doc.page_content, method="percentile", threshold=90)
            for chunk in chunks:
                semantic_chunks.append(chunk)
        
        # Add chunks to vector store
        vector_store.add_texts(
            texts=semantic_chunks,
            metadatas=[{"source": file_path} for _ in semantic_chunks]
        )
        
        print(f"Added {len(semantic_chunks)} semantic chunks from {file_path}")
        return vector_store
    else:
        raise ValueError("The file must be a .docx file.")


if __name__ == "__main__":
    
    docx_files = glob.glob('testing files/*.docx')
    
    if not docx_files:
        print("No .docx files found in 'testing files/' directory.")
    else:
        for file_path in docx_files:
            try:
                load_docs(file_path)
                print(f"✓ Docs from {file_path} added to vector store.")
            except Exception as e:
                print(f"✗ Error processing {file_path}: {e}")
