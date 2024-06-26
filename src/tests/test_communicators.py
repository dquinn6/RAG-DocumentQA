"""Module for testing all communicator classes."""

import pytest

from src.communicators import GPTCommunicator
from src.config import config

API_KEY = config.user_config["ACCESS_TOKEN"]
MODEL_NAME = config.user_config["MODEL_NAME"]


class TestGPT:
    """Class storing all tests related to GPTCommunicator."""

    example_string = "This is a sentence. This is a longer sentence. This is an even longer sentence."

    # Use this fixture for any tests requiring a GPTCommunicator object.
    @pytest.fixture
    def gpt(self):
        return GPTCommunicator(api_key=API_KEY, model_name=MODEL_NAME)

    def test_communication(self, gpt: GPTCommunicator):
        """Test send/receive with GPT."""
        response = gpt.post_prompt("Hi")
        assert isinstance(response, str), "Bad response"

    def test_raises_exception_on_rag_without_vs(self, gpt: GPTCommunicator):
        """Test exception is raised when invoking post_rag_prompt() without initialized vectorstore."""
        with pytest.raises(ValueError):
            gpt.post_rag_prompt("Hi")

    def test_raises_exception_on_bad_name(self: GPTCommunicator):
        """Test exception is raised when giving an unsupported model name to class."""
        with pytest.raises(ValueError):
            _ = GPTCommunicator(api_key=API_KEY, model_name="bad_name")

    def test_token_count(self, gpt: GPTCommunicator):
        """Test token count produces expected values."""
        expected_count = 18
        token_count_example = gpt.count_tokens(self.example_string)
        assert (
            token_count_example == expected_count
        ), f"Failed token count; counted {token_count_example} but expected {expected_count}"

    def test_truncate(self, gpt: GPTCommunicator):
        """Test text is truncated below given token limit."""
        token_limit = 10
        truncated = gpt.truncate_text(self.example_string, token_limit)
        truncated_count = gpt.count_tokens(truncated)
        assert (
            truncated_count <= token_limit
        ), f"Failed truncate; counted {truncated_count} after truncation with token_limit {token_limit}"
