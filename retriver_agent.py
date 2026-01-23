from langchain.agents import create_agent
from langchain_chroma import Chroma
import os 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import dotenv
from langchain.tools import tool

dotenv.load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

chat_model = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key)

embedding_model = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={"device": "cpu", "trust_remote_code": True}
)

vector_store = Chroma(
    embedding_function=embedding_model,
    persist_directory="chroma_db"
)

def rerank_with_keywords(query, results, top_n=3):
    """
    A simple alternative reranking method based on keyword matching and position.
    
    Args:
        query (str): User query
        results (List[Dict]): Initial search results
        top_n (int): Number of results to return after reranking
        
    Returns:
        List[Dict]: Reranked results
    """
    
    keywords = [word.lower() for word in query.split() if len(word) > 3]
    
    scored_results = [] 
    
    for result in results:
        document_text = result["text"].lower() 
        
        
        base_score = result["similarity"] * 0.5
        
        
        keyword_score = 0
        for keyword in keywords:
            if keyword in document_text:
                # Add points for each keyword found
                keyword_score += 0.1
                
                # Add more points if keyword appears near the beginning
                first_position = document_text.find(keyword)
                if first_position < len(document_text) / 4:  # In the first quarter of the text
                    keyword_score += 0.1
                
                # Add points for keyword frequency
                frequency = document_text.count(keyword)
                keyword_score += min(0.05 * frequency, 0.2)  # Cap at 0.2
        
        # Calculate the final score by combining base score and keyword score
        final_score = base_score + keyword_score
        
        # Append the scored result to the list
        scored_results.append({
            "text": result["text"],
            "metadata": result["metadata"],
            "similarity": result["similarity"],
            "relevance_score": final_score
        })
    
    # Sort results by final relevance score in descending order
    reranked_results = sorted(scored_results, key=lambda x: x["relevance_score"], reverse=True)
    
    # Return the top_n results
    return reranked_results[:top_n]

@tool
def retrive_context(query: str) -> str:
    """
    Retrieve relevant documents from the vector store to answer questions.
    Uses semantic search followed by keyword-based reranking.
    """
    # Initial semantic search with k=10 to get candidates
    docs_with_scores = vector_store.similarity_search_with_relevance_scores(query, k=10)
    
    # Convert to format expected by rerank_with_keywords
    results = [
        {
            "text": getattr(doc, "page_content", str(doc)),
            "metadata": doc.metadata if hasattr(doc, "metadata") else {},
            "similarity": score
        }
        for doc, score in docs_with_scores
    ]
    
    # Apply keyword-based reranking to improve relevance
    reranked_results = rerank_with_keywords(query, results, top_n=5)
    
    # Extract and format the reranked results
    texts = [result["text"] for result in reranked_results]
    return "\n\n---\n\n".join(texts)




agent = create_agent(
    model=chat_model,
    tools=[retrive_context],
    system_prompt="""Use the retrive_context tool to answer questions from the database. Quote relevant excerpts from the context. If context doesn't help, say 'I don't know'."""
)

if __name__ == "__main__":
    while True:
        user_input = input("Enter your question (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        try: 
            response = agent.invoke({'messages': [{'role': 'user', 'content': user_input}]})
            print("Tool Message: ", response['messages'][2].content)
            print("Response:", response['messages'][-1].content)
        except Exception as e:
            print("Error during agent invocation:", str(e))

    