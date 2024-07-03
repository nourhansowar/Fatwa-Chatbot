import pandas as pd
import faiss
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import logging
import time
import torch
import numpy as np
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FAISSEmbedding:
    def __init__(self, model_name, token):
        logging.info("Initializing FAISS embedding class")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, trust_remote_code=True)
        
        # Check if the tokenizer has a padding token, if not add one
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, token=token, trust_remote_code=True)
        
        # Resize model embeddings if new tokens were added
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.index = None
        self.embeddings = None
        self.questions = []
        self.answers = []

    def load_csv_data(self, file_path):
        logging.info(f"Loading data from {file_path}")
        data = pd.read_csv(file_path, encoding='utf-8')
        if 'ques' not in data.columns or 'ans' not in data.columns:
            logging.error("CSV file must contain 'ques' and 'ans' columns")
            raise ValueError("CSV file must contain 'ques' and 'ans' columns")
        
        # select first 20000 rows
        data = data[:1000]
        
        self.questions = data['ques'].tolist()
        self.answers = data['ans'].tolist()
        logging.info(f"Data loaded successfully, shape: {data.shape}")
        
    def create_embeddings(self, batch_size=32):
        logging.info("Creating embeddings for the questions")
        
        embeddings_list = []
        for i in range(0, len(self.questions), batch_size):
            batch = self.questions[i:i + batch_size]
            batch = [str(q) for q in batch]  # Ensure all elements are strings
            inputs = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            # Extract the hidden states from the output
            hidden_states = outputs.hidden_states[-1]  # Get the last layer's hidden state
            batch_embeddings = hidden_states.mean(dim=1).cpu().numpy()
            embeddings_list.append(batch_embeddings)
        
        self.embeddings = np.vstack(embeddings_list)
        logging.info("Embeddings created successfully")
        
    def build_index(self, n_list=100):
        logging.info("Building FAISS index")
        dimension = self.embeddings.shape[1]
        quantizer = faiss.IndexFlatL2(dimension)
        self.index = faiss.IndexIVFFlat(quantizer, dimension, n_list)
        self.index.train(self.embeddings)
        self.index.add(self.embeddings)
        logging.info("FAISS index built successfully")

    def save_index(self, index_file_path, metadata_file_path):
        logging.info(f"Saving index to {index_file_path}")
        faiss.write_index(self.index, index_file_path)
        logging.info("Index saved successfully")

        # Save questions and answers
        with open(metadata_file_path, 'wb') as f:
            pickle.dump({'questions': self.questions, 'answers': self.answers}, f)
        logging.info("Metadata saved successfully")

    def load_index(self, index_file_path, metadata_file_path):
        logging.info(f"Loading index from {index_file_path}")
        self.index = faiss.read_index(index_file_path)
        logging.info("Index loaded successfully")

        # Load questions and answers
        with open(metadata_file_path, 'rb') as f:
            metadata = pickle.load(f)
            self.questions = metadata['questions']
            self.answers = metadata['answers']
        logging.info("Metadata loaded successfully")

    def search(self, query, top_k=5):
        logging.info(f"Searching for top {top_k} similar questions for the query: '{query}'")
        inputs = self.tokenizer([query], return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        hidden_states = outputs.hidden_states[-1]  # Get the last layer's hidden state
        query_vector = hidden_states.mean(dim=1).cpu().numpy()
        distances, indices = self.index.search(query_vector, top_k)
        results = [(self.questions[idx], self.answers[idx], distance)
                   for idx, distance in zip(indices[0], distances[0])]
        logging.info(f"Search completed for query: '{query}'")
        return results
