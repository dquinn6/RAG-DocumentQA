"""Module containing classes to process data for vectorstore."""

import logging
import os
from abc import ABC, abstractmethod
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset

from src.communicators import Communicator
from src.utils import manipulate_passages


class DataProcessorError(Exception):
    """Custom error for DataProcessor classes."""

    pass


class DataProcessor(ABC):
    """Base class for data processing into format needed for vectorstore creation."""

    def __init__(self) -> None:
        """Init the object."""
        pass

    @abstractmethod
    def process_text(self, *args, **kwargs) -> Optional[List[str]]:
        """Process raw data into a list of text for vectorstore creation."""
        pass

    @abstractmethod
    def ret_passages_with_pattern(self) -> Optional[List[str]]:
        """Return list of passages containing a text pattern."""
        pass


class WikiTextProcessor(DataProcessor):
    """DataProcessor subclass for processing data into format needed for vectorstore creation."""

    def __init__(
        self,
        dataset_version: str = "wikitext-2-raw-v1",
        split: str = "train",
        communicator: Optional[Communicator] = None,
        verbose: bool = True,
    ) -> None:
        """Init object to process WikiTest dataset.

        Args:
            dataset_version (str, optional): Dataset version matching names on HF. Defaults to "wikitext-2-raw-v1".
            split (str, optional): Split of data to use. Defaults to "train".
            communicator (Optional[Communicator], optional): Communicator object used for counting tokens for trimming data. Defaults to None.
            verbose (bool, optional): Whether to diplay info messages while processing. Defaults to True.
        """
        self.dataset = load_dataset("wikitext", dataset_version)
        self.data = self.dataset[split]["text"]
        self.communicator = communicator  # used for token limiting
        self.verbose = verbose

    def classify_string_type(self, text: str) -> Optional[str]:
        """Classify string as title/header/subheader/content based on the delimiters in the data.

        Args:
            text (str): WikiText raw string.

        Raises:
            DataProcessorError: Raised when method is unsuccessful.

        Returns:
            Optional[str]: Title/header/subheader/content classification; None if fails
        """
        if text == "":
            return "empty"

        # Define delimiters observed in data
        title_delimiter = " = "
        header_delimiter = " = = "
        subheader_delimiter = " = = = "

        def check_by_delimiter(t, delimiter: str) -> bool:
            # When split by the right delimiter, text will be in the form: ['', text, '\n']
            t_split = t.split(delimiter)

            # For titles and headers, we can expect split == 3 and split[-1] == \n
            if len(t_split) == 3 and t_split[-1] == "\n":
                return True
            else:
                return False

        try:
            if check_by_delimiter(text, subheader_delimiter):
                return "subheader"

            elif check_by_delimiter(text, header_delimiter):
                return "header"

            elif check_by_delimiter(text, title_delimiter):
                return "title"

            else:
                return "content"

        except Exception as e:
            raise DataProcessorError(f"Failed to classify string: {e}")
            # logging.error(f"Failed to classify string: {e}")
            # return None

    def transform_list_into_passages(self, text_list: List[str]) -> List[str]:
        """Transform a list of newline strings into a list of full passage strings.

        Args:
            text_list (List[str]): List of newline strings from dataset.

        Raises:
            DataProcessorError: Raised when method is unsuccessful.

        Returns:
            List[str]: List of passages
        """
        try:
            # Store counts of titles, headers, etc. for double check
            text_type = list(map(lambda t: self.classify_string_type(t), text_list))
            type_counts = Counter(text_type)

            # Get indicies of strings in list classified as titles
            title_idx = np.array([i for i, v in enumerate(text_type) if v == "title"])
            title_idx = np.append(title_idx, len(text_list))  # Append for last passage
            title_idx_pairs = np.column_stack((title_idx[:-1], title_idx[1:]))

            # Slice between title indicies to form full passage
            passages = []
            for idx_pair in title_idx_pairs:
                start_i, end_i = idx_pair[0], idx_pair[1]
                passage = "\n".join(text_list[start_i:end_i])
                passages.append(passage)

            assert (
                len(passages) == type_counts["title"]
            ), "Passage count should match number of titles"

        except Exception as e:
            raise DataProcessorError(f"Failed to transform list into passages: {e}")
            # logging.error(f"Failed to transform list into passages: {e}")
            # return None

        return passages

    def process_text(
        self,
        token_limit: Optional[int] = None,
        save_path: Optional[str] = None,
        save_filename: str = "processed_data",
        manipulate_pattern: Optional[List[Tuple[str, str]]] = None,
    ) -> List[str]:
        """Process the initialized data from a list of text to a list of passages.

        These passages can also be manipulated by supplying a list of tuple pairs (search, replace).

        Args:
            token_limit (Optional[int], optional): Specify token limit to filter passages. Defaults to None.
            save_path (Optional[str], optional): Path to save to. Defaults to None.
            save_filename (str, optional): Filename for output csv; extension is appended if not included. Defaults to "processed_data".
            manipulate_pattern (Optional[List[Tuple[str, str]]], optional): An optional list of tuple str pairs to manipulate passages. Defaults to None.

        Raises:
            DataProcessorError: Raised when method is unsuccessful.

        Returns:
            List[str]: Processed list of passages.
        """
        if save_path is not None:
            if not os.path.exists(save_path):
                os.mkdir(save_path)

        # Create list of passages from raw data
        passages = self.transform_list_into_passages(self.data)

        if self.verbose:
            logging.info(f"{len(passages)} passages created. ")

        try:
            # Filter out passages if token limit was set
            if token_limit:
                # Exit if communicator was never initialized
                if self.communicator is None:
                    logging.error(
                        "Cannot limit tokens without providing a Communicator object. "
                    )
                    return None

                # Filter passages
                passages = [
                    p
                    for p in passages
                    if self.communicator.count_tokens(p) <= token_limit
                ]

                if self.verbose:
                    logging.info(
                        f"{len(passages)} passages remaining after limiting tokens"
                    )
                    largest_passage_size = np.max(
                        list(map(lambda p: self.communicator.count_tokens(p), passages))
                    )
                    logging.info(
                        f"largest passage after trim is {largest_passage_size} tokens"
                    )

            if manipulate_pattern:
                for pattern in manipulate_pattern:
                    passages = manipulate_passages(
                        passages, pattern, verbose=self.verbose
                    )

            write_df = pd.DataFrame()
            write_df["text"] = passages

            if save_path is not None:
                save_to = save_path + save_filename
                if save_to[-4:] != ".csv":
                    save_to += ".csv"

                write_df.to_csv(save_to)

                if self.verbose:
                    logging.info(f"Processed data saved to: {save_to}")

        except Exception as e:
            raise DataProcessorError(f"Failed to process data:  {e}")
            # logging.error(f"Failed to process data: {e}")
            # return None

        return passages

    def ret_passages_with_pattern(self, pattern: str) -> List[str]:
        """Return list of passages containing a search pattern.

        Args:
            pattern (str): Pattern to search for in documents.

        Returns:
            List[str]: Passages with matching pattern.
        """
        passages = self.transform_list_into_passages(self.data)
        return [p for p in passages if pattern in p]
