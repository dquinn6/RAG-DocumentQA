"""Module for testing all vectorstore classes."""

import pytest
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.communicators import GPTCommunicator
from src.config import config
from src.data_processors import WikiTextProcessor
from src.vectorstore_handlers import (
    LangchainVectorstoreFAISS,
    VectorstoreHandler,
)

API_KEY = config.user_config["ACCESS_TOKEN"]
SAVE_PATH = config.user_config["SAVE_PATH"]
MODEL_NAME = config.user_config["MODEL_NAME"]


class TestVectorstoreHandlerFAISS:
    """Class storing all tests for FAISS vectorstore handler."""

    test_filename = "test_data.csv"
    test_vs_path = SAVE_PATH + "test/"

    # Define an example string to use in these tests; needs to a string found in at least one of the documents.
    example_string = "New York State Route"

    # Use this fixture for any tests requiring vectorstore
    @pytest.fixture
    def vs(self):
        communicator = GPTCommunicator(api_key=API_KEY, model_name=MODEL_NAME)
        processor = WikiTextProcessor(
            dataset_version="wikitext-2-raw-v1",
            split="train",
            communicator=communicator,
            verbose=False,
        )
        _ = processor.process_text(
            token_limit=500,  # make small for quick testing
            save_path=SAVE_PATH,
            save_filename=self.test_filename,
        )
        vs = LangchainVectorstoreFAISS(
            embedding_type=HuggingFaceEmbeddings(),
            processed_csv_path=SAVE_PATH + self.test_filename,
            verbose=False,
        )
        return vs

    # Use this fixture for any tests requiring communicator
    @pytest.fixture
    def communicator(self):
        return GPTCommunicator(api_key=API_KEY, model_name=MODEL_NAME)

    def test_create_and_retrieve(self, vs: VectorstoreHandler):
        """Test vectorstore is created properly and can retrieve docs."""
        n_retrieve = 2
        vs.create_local_vectorstore(
            save_path=self.test_vs_path, force_create=True
        )
        vs.create_retriever(search_kwargs={"k": n_retrieve})
        retrieved = vs.retrieve_top_documents(self.example_string)

        assert (
            len(retrieved) == n_retrieve
        ), "Failed to retrieve correct number of documents"

        # Some strings may not always be in retrieved docs, but should be for this test case
        assert all(
            [self.example_string in ret for ret in retrieved]
        ), "Bad retrieved docuemnts"

    def test_load_and_retrieve(self, vs: VectorstoreHandler):
        """Test vectorstore is loaded properly and can retrieve docs."""
        n_retrieve = 2
        vs.load_local_vectorstore(load_path=self.test_vs_path)
        vs.create_retriever(search_kwargs={"k": n_retrieve})
        retrieved = vs.retrieve_top_documents(self.example_string)

        assert (
            len(retrieved) == n_retrieve
        ), "Failed to retrieve correct number of documents"

        # some strings may not always be in retrieved docs, but should be for this test case
        assert all(
            [self.example_string in ret for ret in retrieved]
        ), "Bad retrieved docuemnts"

    def test_text_chunking(
        self, vs: VectorstoreHandler, communicator: GPTCommunicator
    ):
        "Test documents are chunked properly to given size."
        chunk_size = 250
        vs.chunk_data(chunk_size=chunk_size, chunk_overlap=0)
        assert (
            max(
                [
                    communicator.count_tokens(vs.data[i].page_content)
                    for i in range(len(vs.data))
                ]
            )
            < chunk_size
        ), "Bad text chunking"
