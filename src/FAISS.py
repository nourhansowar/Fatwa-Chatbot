import pandas as pd
import faiss
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import logging
import time
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FAISSEmbedding:
    def __init__(self, model_name, token):
        logging.info("Initializing FAISS embedding class")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        
        # Check if the tokenizer has a padding token, if not add one
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        self.model = AutoModel.from_pretrained(model_name, token=token)
        
        # Resize model embeddings if new tokens were added
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.index = None
        self.data = None
        self.embeddings = None

    def load_csv_data(self, file_path):
        logging.info(f"Loading data from {file_path}")
        self.data = pd.read_csv(file_path, encoding='utf-8')
        if 'ques' not in self.data.columns or 'ans' not in self.data.columns:
            logging.error("CSV file must contain 'question' and 'answer' columns")
            raise ValueError("CSV file must contain 'question' and 'answer' columns")
        logging.info(f"Data loaded successfully, shape: {self.data.shape}")
        
        # print first 5 rows of the data
        print(self.data.head())
        
    def load_json_data(self, file_path):
        #TODO: Implement this method
        pass

    def create_embeddings(self):
        logging.info("Creating embeddings for the questions")
        questions = self.data['ques'].tolist()
        
        # Convert all questions to strings
        questions = [str(question) for question in questions]
    
        if not all(isinstance(question, str) for question in questions):
            logging.error("All questions must be strings")
            raise ValueError("All questions must be strings")
        
        inputs = self.tokenizer(questions, return_tensors='pt', padding=True, truncation=False)
        with torch.no_grad():
            outputs = self.model(**inputs)
        self.embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        logging.info("Embeddings created successfully")
        
    def build_index(self, n_list=100):
        logging.info("Building FAISS index")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        logging.info("FAISS index built successfully")

    def save_index(self, index_file_path):
        logging.info(f"Saving index to {index_file_path}")
        faiss.write_index(self.index, index_file_path)
        logging.info("Index saved successfully")

    def load_index(self, index_file_path):
        logging.info(f"Loading index from {index_file_path}")
        self.index = faiss.read_index(index_file_path)
        logging.info("Index loaded successfully")

    def search(self, query, top_k=5):
        logging.info(f"Searching for top {top_k} similar questions for the query: '{query}'")
        inputs = self.tokenizer(query, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        query_vector = outputs.last_hidden_state.mean(dim=1).numpy()
        distances, indices = self.index.search(query_vector, top_k)
        results = [(self.data.iloc[idx]['ques'], self.data.iloc[idx]['ans'], distance)
                   for idx, distance in zip(indices[0], distances[0])]
        logging.info(f"Search completed for query: '{query}'")
        return results