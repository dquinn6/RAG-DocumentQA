from datasets import load_dataset
import logging
from typing import Optional, List, Tuple
from collections import Counter
import numpy as np
import pandas as pd
import os
from src.communicators import Communicator
from src.utils import manipulate_passages
from abc import ABC, abstractmethod


class DataProcessor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def process_text(self):
        """ A method to process raw data into a list of text for vectorstore creation. """
        pass

    @abstractmethod
    def ret_passages_with_pattern(self):
        """ A method to return list of passages containing a text pattern. """
        pass


class WikiTextProcessor(DataProcessor):

    def __init__(
            self, 
            dataset_version = "wikitext-2-raw-v1", 
            split="train", 
            communicator: Optional[Communicator] = None,
            verbose: bool=True,
        ):

        self.dataset = load_dataset("wikitext", dataset_version)
        self.data = self.dataset[split]["text"]
        self.communicator = communicator # used for token limiting 
        self.verbose = verbose


    def classify_string_type(self, text: str) -> Optional[str]:
        # define a function to classify string as title/header/content based on the delimiters we saw above

        if text == '':
            return "empty"
        
        title_delimiter = " = "
        header_delimiter = " = = "
        subheader_delimiter = " = = = "

        def check_by_delimiter(t, delimiter: str) -> bool:
            # when split by the right delimiter, text will be in the form: ['', text, '\n']
            t_split = t.split(delimiter)

            # for titles and headers, we can expect split == 3 and split[-1] == \n
            if len(t_split) == 3 and t_split[-1] == '\n':
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
            logging.error(f"Failed to classify string: {e}")
            return None 
        

    def transform_list_into_passages(self, text_list: str) -> Optional[List[str]]:

        try:
            text_type = list(map(lambda t: self.classify_string_type(t), text_list))

            type_counts = Counter(text_type) # dict storing counts of titles, headers, etc.

            title_idx = np.array([i for i,v in enumerate(text_type) if v == "title"])
            title_idx = np.append(title_idx, len(text_list)) # append for last passage
            title_idx_pairs = np.column_stack((title_idx[:-1], title_idx[1:]))

            passages = []

            for idx_pair in title_idx_pairs:
                start_i, end_i = idx_pair[0], idx_pair[1]
                passage = "\n".join(text_list[start_i:end_i])
                passages.append(passage)

            assert len(passages) == type_counts["title"], "Passage count should match number of titles"

        except Exception as e:
            logging.error(f"Failed to transform list into passages: {e}")
            return None

        return passages
    
    def process_text(
            self, 
            token_limit: Optional[int] = None,
            save_path: str = os.getcwd() + os.sep,
            save_filename: str = "processed_data",
            manipulate_pattern: Optional[List[Tuple[str, str]]] = None,
        ):

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        passages = self.transform_list_into_passages(self.data)
        
        if self.verbose:
            logging.info(f"{len(passages)} passages created. ")

        try:
            if token_limit:
                # exit if communicator was never initialized
                if self.communicator == None:
                    logging.error("Cannot limit tokens without providing a Communicator object. ")
                    return None
                
                passage_token_counts = list(map(lambda p: self.communicator.count_tokens(p), passages))
                
                valid_idx = [i for i,v in enumerate(passage_token_counts) if v <= token_limit]
                passages = [v for i,v in enumerate(passages) if i in valid_idx]

                if self.verbose:
                    logging.info(f"{len(passages)} passages remaining after limiting tokens")
                    largest_passage_size = np.max(list(map(lambda p: self.communicator.count_tokens(p), passages)))
                    logging.info(f"largest passage after trim is {largest_passage_size} tokens")

            if manipulate_pattern:
                for pattern in manipulate_pattern:
                    passages = manipulate_passages(passages, pattern, verbose=self.verbose)

            write_df = pd.DataFrame()
            write_df["text"] = passages

            save_to = save_path + save_filename
            if save_to[-4:] != ".csv":
                save_to += ".csv"

            write_df.to_csv(save_to)

            if self.verbose:
                logging.info(f"Processed data saved to: {save_to}")

        except Exception as e:
            logging.error(f"Failed to process data: {e}")
            return None
        
        return passages
    
    def ret_passages_with_pattern(self, pattern: str):
        passages = self.transform_list_into_passages(self.data)
        return [p for p in passages if pattern in p]
