import pytest
from src.communicators import GPTCommunicator
from src.data_processors import WikiTextProcessor
from src.config import config

API_KEY = config.user_config["ACCESS_TOKEN"]
MODEL_NAME = config.user_config["MODEL_NAME"]

class TestWikiTextProcessor:

    example_string = "New York State Route"

    @pytest.fixture
    def processor(self):
        communicator = GPTCommunicator(api_key=API_KEY, model_name=MODEL_NAME)
        processor = WikiTextProcessor(
            dataset_version = "wikitext-2-raw-v1", 
            split = "train", 
            communicator = communicator,
            verbose = False
        )
        return processor

    @pytest.mark.parametrize("string_with_delimiter,expected", [
        (" = Title = \n", "title"),
        (" = = Header = = \n", "header"),
        (" = = = Subheader = = = \n", "subheader"),
        ("Content \n", "content")
    ])
    def test_string_classify(self, processor, string_with_delimiter, expected):
        assert processor.classify_string_type(string_with_delimiter) == expected, "Bad string classification"

    def test_pattern_match(self, processor):
        passage_matches = processor.ret_passages_with_pattern(self.example_string)
        assert len(passage_matches) > 0, "Bad test; no passages with this example_string found"
        assert all([self.example_string in passage for passage in passage_matches])

    def test_passage_manipulation(self, processor):

        replace_with = "Some other string"

        passages = processor.process_text(
            token_limit = 500, 
            save_path = None,
            manipulate_pattern = [(self.example_string, replace_with)],
        )
        assert len(passages) > 0, "Bad search; no passages under this token limit"
        
        filter_passages = [p for p in passages if replace_with in p]
        assert all([self.example_string not in p for p in filter_passages]), "Bad manipulation; all patterns were not replaced"
