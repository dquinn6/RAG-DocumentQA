"""Module containing classes to handle the creation, loading, and retrieval from our vectorstore of given documents."""

import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings.embeddings import Embeddings
from tqdm import tqdm


class VectorstoreError(Exception):
    """Custom error for vectorstore handlers."""

    pass


class VectorstoreHandler(ABC):
    """Base class for handling vector store functionality."""

    def __init__(self):
        """Init the object."""
        pass

    @abstractmethod
    def create_local_vectorstore(self):
        """Generate a vectorstore index from a set of docs and writes to local file."""
        pass

    @abstractmethod
    def load_local_vectorstore(self):
        """Load the currently saved local vectorstore index."""
        pass

    @abstractmethod
    def retrieve_top_documents(self):
        """Retrieve top document matches for RAG prompting."""
        pass


class LangchainVectorstoreFAISS(VectorstoreHandler):
    """Langchain implementation of FAISS vectorstore index."""

    def __init__(
        self,
        embedding_type: Embeddings = HuggingFaceEmbeddings,
        processed_csv_path: str = "processed_data.csv",
        verbose: bool = True,
    ) -> None:
        """Initialize the FAISS vectorstore object.

        Args:
            embedding_type (langchain_community.embeddings, optional): Type of embedding algorithm to use. Defaults to HuggingFaceEmbeddings.
            processed_csv_path (str, optional): Path to processed data to use for vectorstore. Defaults to "processed_data.csv".
            verbose (bool, optional): Display logging info messages. Defaults to True.
        """
        super().__init__()
        self.data = self.load_csv_file(processed_csv_path)
        self.embedding_type = embedding_type
        self.verbose = verbose
        self.vectorstore = (
            None  # invoke create/load_local_vectorstore() to set
        )
        self.retriever = None  # invoke create_retriever to set

        if self.verbose:
            logging.info(
                "Vectorstore and retriever must be set using the class methods."
            )

    def load_csv_file(
        self, file_path: str, shuffle: bool = True, seed: Optional[int] = None
    ) -> Document:
        """Load the CSV data using Langchain's loader.

        Args:
            file_path (str): Path to processed CSV data.
            shuffle (bool, optional): Shuffle the data. Defaults to True.
            seed (Optional[int, optional): Random seed for shuffle. Defaults to None.

        Returns:
            Document: Data from CSV loader.
        """
        # keeping this function within this class since it uses the langchain loader
        try:
            loader = CSVLoader(
                file_path=file_path,
                encoding="utf-8",
                csv_args={"delimiter": ","},
            )
            csv_data = loader.load()
            if shuffle:
                random.seed(seed)
                random.shuffle(csv_data)

        except Exception as e:
            raise VectorstoreError(f"Failed to load csv data: {e}")

        return csv_data

    def chunk_data(
        self, chunk_size: int = 2048, chunk_overlap: int = 50
    ) -> None:
        """Chunk the data to specified chunk size using RecursiveCharacterTextSplitter.

        Args:
            chunk_size (int, optional): Token size to chunk data to. Defaults to 2048.
            chunk_overlap (int, optional): Token overlap between chunks. Defaults to 50.
        """
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            self.data = text_splitter.split_documents(self.data)
            if self.verbose:
                logging.info(f"Data chunked to size {chunk_size}")

        except Exception as e:
            logging.error(f"Failed to chunk data: {e}")

    def create_local_vectorstore(
        self,
        save_path: str,
        force_create: bool = False,
        callback: Optional[Callable] = None,
    ) -> None:
        """Create a new local vectorstore index.

        Args:
            save_path (str): Path to save vectorstore index to.
            force_create (bool, optional): Forces creation of vectostore and avoid 'already found; load instead?' check. Defaults to False.
            callback (Optional[Callable], optional): Optional callback function. Defaults to None.

        Raises:
            ValueError: Raised when given invalid input for 'already found; load instead?' check.
            VectorstoreError: Raised when vectorstore creation was unsuccessful.
        """
        if os.path.exists(save_path) and not force_create:
            overwrite_saved = input(
                f"Vectorstore already found at {save_path}; overwrite? [y/n]: "
            ).lower()
            if overwrite_saved not in ["y", "n"]:
                raise ValueError(
                    "Invalid input; try again with 'y' for yes or 'n' for no."
                )

            if overwrite_saved == "n":
                load_instead = input("Load instead [y/n]: ").lower()
                if load_instead not in ["y", "n"]:
                    raise ValueError(
                        "Invalid input; try again with 'y' for yes or 'n' for no."
                    )

                if load_instead == "y":
                    self.load_local_vectorstore(load_path=save_path)
                else:
                    if self.verbose:
                        logging.info(
                            "Vectorstore not overwritten; aborting... "
                        )

        else:
            logging.info(f"Creating a new local vectorstore at: {save_path}")
            try:
                # No built in progress bar from their API; using this workaround shared at: https://stackoverflow.com/questions/77836174/how-can-i-add-a-progress-bar-status-when-creating-a-vector-store-with-langchain
                total_len = len(self.data)

                # Below function is equivalent to embedding each piece of text, zipping text and embeddings as pairs, and creating index from these pairs
                with tqdm(
                    total=total_len, desc="Processing documents"
                ) as progress_bar:
                    for d in range(total_len):
                        doc = self.data[d]
                        # If already init, add next doc
                        if self.vectorstore:
                            self.vectorstore.add_documents([doc])
                        else:  # Init vectorstore with first doc
                            self.vectorstore = FAISS.from_documents(
                                [doc], self.embedding_type
                            )

                        # Update progress bar and optional callback for higher level (UI) progress bars.
                        progress_bar.update(1)
                        if callback:
                            callback(d / total_len, d)

                self.vectorstore.save_local(save_path)
                if self.verbose:
                    logging.info(
                        f"Vectorstore successfully set and saved to {save_path}"
                    )

            except Exception as e:
                raise VectorstoreError(f"Failed to create vectorstore: {e}")

    def load_local_vectorstore(self, load_path: str) -> None:
        """Load previously generated local vectorstore index.

        Args:
            load_path (str): Path to local vectorstore.

        Raises:
            FileNotFoundError: Raised when index cannot be found at load_path.
            VectorstoreError: Raised when vectorstore load was unsuccessful.
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(
                f"Failed to find a saved vectorstore at {load_path}; please ensure save_path points to correct location."
            )

        try:
            self.vectorstore = FAISS.load_local(
                load_path,
                self.embedding_type,
                allow_dangerous_deserialization=True,
            )
            if self.verbose:
                logging.info("Vectorstore loaded successfully.")

        except Exception as e:
            raise VectorstoreError(f"Failed to load vectorstore: {e}")

    def create_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: Optional[dict] = None,
    ) -> None:
        """Create document retriever object for vectorstore.

        Args:
            search_type (str, optional): Algorithm to use when searching for related documents. Defaults to "similarity".
            search_kwargs (Optional[dict], optional): Retriever kwargs, which can include 'k' for number of retrieved docs. Defaults to None.

        Raises:
            VectorstoreError: Raised when vectorstore was initialized but never loaded/created.
            ValueError: Raised when given an invalid arg for search_type.
            VectorstoreError: Raised when retriever creation was unsuccessful.
        """
        if self.vectorstore is None:
            raise VectorstoreError(
                "Vectorstore not set; create or load a vectorstore using class method first."
            )

        search_types = ["similarity", "mmr", "similarity_score_threshold"]
        if search_type not in search_types:
            raise ValueError(
                f"Invalid arg for search_type; valid args include: {search_types}"
            )

        try:
            self.retriever = self.vectorstore.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs,
            )
            if self.verbose:
                logging.info("Retriever successfully set")

        except Exception as e:
            raise VectorstoreError(f"Failed to create retriever: {e}")
            # logging.error(f"Failed to create retriever: {e}")

    def retrieve_top_documents(self, query: str) -> List[str]:
        """Use retriever to fetch top k matching documents for RAG.

        Args:
            query (str): Query to search documents against.

        Raises:
            VectorstoreError: Raised when retriever was never created.

        Returns:
            Optional[List[str]]: List of retrieved documents for RAG.
        """
        if self.retriever is None:
            raise VectorstoreError(
                "Retriver not set; create a retriever using class method first"
            )

        try:
            retrieved_docs = self.retriever.get_relevant_documents(query)
        except Exception as e:
            raise VectorstoreError(f"Failed to retrive documents: {e}")

        return [
            retrieved_docs[i].page_content for i in range(len(retrieved_docs))
        ]
