"""Module for testing all data processor classes."""

import pytest

from src.communicators import GPTCommunicator
from src.config import config
from src.data_processors import WikiTextProcessor

API_KEY = config.user_config["ACCESS_TOKEN"]
MODEL_NAME = config.user_config["MODEL_NAME"]


class TestWikiTextProcessor:
    """Class storing all tests related to WikiText DataProcessor."""

    # Define an example string to use in these tests; needs to a string found in at least one of the documents.
    example_string = "New York State Route"

    # Use this fixture for any tests requiring a WikiText Processor object.
    @pytest.fixture
    def processor(self):
        communicator = GPTCommunicator(api_key=API_KEY, model_name=MODEL_NAME)
        processor = WikiTextProcessor(
            dataset_version="wikitext-2-raw-v1",
            split="train",
            communicator=communicator,
            verbose=False,
        )
        return processor

    @pytest.mark.parametrize(
        "string_with_delimiter,expected",
        [
            (" = Title = \n", "title"),
            (" = = Header = = \n", "header"),
            (" = = = Subheader = = = \n", "subheader"),
            ("Content \n", "content"),
        ],
    )
    def test_string_classify(
        self, processor: WikiTextProcessor, string_with_delimiter: str, expected: str
    ):
        """Test processor correctly classifies above strings based on the delimiter format observed in the dataset."""
        assert (
            processor.classify_string_type(string_with_delimiter) == expected
        ), "Bad string classification"

    def test_pattern_match(self, processor: WikiTextProcessor):
        """Test passages returned from pattern matching actually contain the specified str."""
        passage_matches = processor.ret_passages_with_pattern(self.example_string)
        assert (
            len(passage_matches) > 0
        ), "Bad test; no passages with this example_string found"
        assert all([self.example_string in passage for passage in passage_matches])

    def test_passage_manipulation(self, processor: WikiTextProcessor):
        """Test passages are properly manipulated some another str."""

        # Define a replacement str different from example str
        replace_with = "Some other string"

        passages = processor.process_text(
            token_limit=500,  # limit token size for quicker test
            save_path=None,
            manipulate_pattern=[(self.example_string, replace_with)],
        )
        assert len(passages) > 0, "Bad search; no passages under this token limit"

        filter_passages = [p for p in passages if replace_with in p]
        assert all(
            [self.example_string not in p for p in filter_passages]
        ), "Bad manipulation; all patterns were not replaced"
