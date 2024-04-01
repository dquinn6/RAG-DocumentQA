""" Classes to handle the creation, loading, and retrieval from our vectorstore of given documents. """

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader

from langchain_core.documents import Document
import random
from typing import Optional, List
from abc import ABC, abstractmethod

from tqdm import tqdm
import logging
import os

class VectorstoreHandler(ABC):
    """ Base class for handling vector store functionality. 
    Basic implementations will need ways to create/load a vectorstore and retrieve documents.
    """
    def __init__(self):
        pass

    @abstractmethod
    def create_local_vectorstore(self):
        """ A method to generate a vectorstore index from a set of docs and write to local file. """
        pass

    @abstractmethod
    def load_local_vectorstore(self):
        """ A method to load the currently saved local vectorstore index. """
        pass

    @abstractmethod
    def retrieve_top_documents(self):
        """ A method to retrieve top document matches for RAG prompting. """
        pass


class LangchainVectorstore(VectorstoreHandler):
    def __init__(self, embedding_type, processed_csv_path, verbose=True):
        super().__init__()
        self.data = self.load_csv_file(processed_csv_path)
        self.embedding_type = embedding_type
        self.verbose = verbose
        self.vectorstore = None # invoke create/load_local_vectorstore() to set
        self.retriever = None # invoke create_retriever to set

        if self.verbose:
            logging.info("Vectorstore and retriever must be set using the class methods.")

    def load_csv_file(self, file_path: str, shuffle=True, seed=None) -> Optional[Document]:
        # keeping this function within this class since it uses the langchain loader
        try:
            loader = CSVLoader(
                file_path=file_path, encoding="utf-8", csv_args={"delimiter": ","}
            )
            csv_data = loader.load()
            if shuffle:
                random.seed(seed)
                random.shuffle(csv_data)

        except Exception as e:
            logging.error(f"Failed to load csv data: {e}")
            return None

        return csv_data

    def chunk_data(self, chunk_size: int = 2048, chunk_overlap: int = 50) -> None:

        try:
            text_splitter = CharacterTextSplitter(
                        chunk_size=chunk_size, chunk_overlap=chunk_overlap
                    )
            self.data = text_splitter.split_documents(self.data)
            if self.verbose:
                logging.info(f"Data chunked to size {chunk_size}")
    
        except Exception as e:
            logging.error(f"Failed to chunk data: {e}")


    def create_local_vectorstore(self, save_path: str, force_create = False, callback=None) -> None:

        if os.path.exists(save_path) and not force_create:
            overwrite_saved = input(f"Vectorstore already found at {save_path}; overwrite? [y/n]: ").lower()
            if overwrite_saved not in ["y", "n"]:
                raise ValueError("Invalid input; try again with 'y' for yes or 'n' for no.")
            
            if overwrite_saved == "n":
                load_instead = input(f"Load instead [y/n]: ").lower()
                if load_instead not in ["y", "n"]:
                    raise ValueError("Invalid input; try again with 'y' for yes or 'n' for no.")
                
                if load_instead == "y":
                    self.load_local_vectorstore(load_path=save_path)
                else:
                    if self.verbose:
                        logging.info("Vectorstore not overwritten; aborting... ")

                return None # break out of function

        logging.info(f"Creating a new local vectorstore at: {save_path}")
        try:
            # no built in progress bar from their API; using this workaround shared at: https://stackoverflow.com/questions/77836174/how-can-i-add-a-progress-bar-status-when-creating-a-vector-store-with-langchain
            
            total_len = len(self.data)

            with tqdm(total=total_len, desc="Processing documents") as progress_bar:
                for d in range(total_len):
                    doc = self.data[d]
                    if self.vectorstore:
                        self.vectorstore.add_documents([doc])
                    else: # init 
                        self.vectorstore = FAISS.from_documents([doc], self.embedding_type)

                    progress_bar.update(1)
                    if callback:
                        callback(d/total_len, d)


            #self.vectorstore = FAISS.from_documents(self.data, self.embedding_type)
            # above function is equivalent to embedding each piece of text, zipping text and embeddings as pairs, and creating index from these pairs
            self.vectorstore.save_local(save_path)
            if self.verbose:
                logging.info(f"Vectorstore successfully set and saved to {save_path}")
        
        except Exception as e:
            logging.error(f"Failed to create vectorstore: {e}")
        
        finally:
            return None # not needed, but including a final return since we used one for a conditional abort
            

    def load_local_vectorstore(self, load_path: str) -> None:

        if not os.path.exists(load_path):
            raise ValueError(f"Failed to find a saved vectorstore at {load_path}; please ensure save_path points to correct location.")

        try:
            self.vectorstore = FAISS.load_local(load_path, self.embedding_type, allow_dangerous_deserialization=True)
            if self.verbose:
                logging.info("Vectorstore loaded successfully.")

        except Exception as e:
            logging.error(f"Failed to load vectorstore: {e}")

    def create_retriever(self, search_type: str = "similarity", search_kwargs: dict = {}) -> None:

        if self.vectorstore is None:
            raise ValueError("Vectorstore not set; create or load a vectorstore using class method first.")

        search_types = ["similarity", "mmr", "similarity_score_threshold"]
        if search_type not in search_types:
            raise ValueError(f"Invalid arg for search_type; valid args include: {search_types}")
        
        try:
            self.retriever = self.vectorstore.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs,
            )
            if self.verbose:
                logging.info(f"Retriever successfully set")

        except Exception as e:
            logging.error(f"Failed to create retriever: {e}")

        
    def retrieve_top_documents(self, query: str) -> Optional[List[str]]:
        if self.retriever is None:
            raise ValueError("Retriver not set; create a retriever using class method first")
        
        try:
            retrieved_docs = self.retriever.get_relevant_documents(query)
        except Exception as e:
            logging.error(f"Failed to retrive documents: {e}")
            return None

        return [retrieved_docs[i].page_content for i in range(len(retrieved_docs))]