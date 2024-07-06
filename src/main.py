from FAISS import *
from tokens import TOKEN_API
csv_file_path = '../data/faq_data.csv'

if __name__ == "__main__":
    
    # Use the model from the CohereForAI collection
    # model_name = "CohereForAI/aya-23-35B"
    model_name = "aubmindlab/aragpt2-large"
    # model_name = 'core42/jais-13b'
    faiss_embedding = FAISSEmbedding(model_name, TOKEN_API)

    # Load data from a CSV file
    start_time = time.time()
    faiss_embedding.load_csv_data(csv_file_path)
    logging.info(f"Loading data took {time.time() - start_time:.2f} seconds")

    # Create embeddings for the questions
    start_time = time.time()
    faiss_embedding.create_embeddings(batch_size=32)
    logging.info(f"Creating embeddings took {time.time() - start_time:.2f} seconds")

    # Build the FAISS index
    start_time = time.time()
    faiss_embedding.build_index()
    logging.info(f"Building the index took {time.time() - start_time:.2f} seconds")

    # Save the index to a file
    start_time = time.time()
    faiss_embedding.save_index('../data/faiss_index_file.index', '../data/metadata.pkl')
    logging.info(f"Saving the index took {time.time() - start_time:.2f} seconds")