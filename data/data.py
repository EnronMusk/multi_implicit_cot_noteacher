import pandas as pd
import secrets as s

from torch.utils.data import Dataset
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

import torch
import copy


def extractAnswer(text):
    split_pattern = '####'
    if split_pattern not in text:
        return text.strip().replace(',', '')
    else:
        _, ans = text.strip().split('####', 1)
        ans = '####' + ans
        ans = ans.strip().replace(',', '')
        return ans

def extractCoT(text):
    split_pattern = '####'
    if split_pattern not in text:
        return None
    else:
        cot, _ = text.strip().split('####', 1)
        cot = cot.strip()
        return cot

class DatasetHandler(Dataset):
    """
    Can generate both train and test datasets.
    Can also generate tokenzied data for model training.
    @type is the type of the dataset, train or test
    """
    
    def __init__(self, parent_path : str, max_len : int, type : str):
        super().__init__()
        self.path = parent_path
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.max_len = max_len
        self.type = type

    def generateDataset(self, size: int = 1, a : int = None, b : int = None, c : int = None, d : int = None) -> None:
        '''
        Generates the dataset of a given size and type.
        @a,b,c,d are for a custom entry, if we want a custom prediction.
        '''
        data = []
        delimiter_input = '||'
        delimter_output = ' #### '
        delmiter_problem = ' $ '

        def __generateProblem(i : int, j : int) -> None:
            if i is None: i = s.randbelow(100)
            if j is None: j = s.randbelow(100)

            j_digit_1 = j // 10
            j_digit_2 = j % 10

            z1 = i*j_digit_2 #First intermediate sum.
            z2 = j_digit_1*i*10 #Second intermediate sum.

            prod = i*j
            
            assert z1 + z2 == prod

            #Convert all numbers to strings then reverse them. Also ensure appropriate length using zfill. z is our CoT.
            i = str(i).zfill(2)
            j = str(j).zfill(2)
            z1 = str(z1).zfill(3)
            z2 = str(z2).zfill(4)
            prod = str(prod).zfill(4)

            input = __addSpaces((i[::-1] + "*" + j[::-1]))
            z = __addSpaces(z1[::-1] + "+" + z2[::-1])
            prod = __addSpaces(prod[::-1])

            return [input, z, prod]

        def __addSpaces(str : str) -> str:
            return " ".join(str)
        
        #Generate entries here by assembling each problem into a full entry.
        for _ in range(size):
            problem_1 = __generateProblem(a,b)
            problem_2 = __generateProblem(c,d)

            entry = problem_1[0] + delmiter_problem + problem_2[0] + delimiter_input + problem_1[1] + delmiter_problem + problem_2[1] + delimter_output + problem_1[2] + delmiter_problem + problem_2[2]

            data.append(entry)

        #Save the dataset
        data = pd.DataFrame(data)
        file_path = self.path + r"\data\raw_" + self.type + r"_dataset.txt"
        data.to_csv(file_path, index = False, header = False)
        print(f'Generated raw {self.type} dataset saved at {file_path} of size {size}.')

        self.__tokenizeDataset() #Automatically tokenize the dataset.

    def __tokenizeDataset(self) -> Dataset:
        '''
        Creates tokenized labels and features to be used in our model.
        Creates the features for a given dataset.
        '''
        file_path = self.path + r"\data\raw_" + self.type + r"_dataset.txt"
        tokenizer = self.tokenizer

        print (f'Creating tokenized features from dataset file at {file_path}')
        bos_tok = tokenizer.bos_token
        eos_tok = tokenizer.eos_token

        with open(file_path, encoding="utf-8") as f:
            lines = [line.split('||') for line in f.read().splitlines() if (len(line) > 0 and not line.isspace()
                                                                             and len(line.split('||')) ==2 )]
        src_lines, tgt_lines = list(zip(*lines))
        src_lines = list(src_lines)
        tgt_lines = list(tgt_lines)

        edited_sents_cot = []
        edited_sents_only = []
        edited_sents_all = []
        edited_sents_nocot = []
        for src, tgt in zip(src_lines, tgt_lines):
            #import pdb; pdb.set_trace()
            ans = extractAnswer(tgt)
            cot = extractCoT(tgt)
            sent = ' {} {} '.format(src, bos_tok) + cot + ' {}'.format(eos_tok)
            edited_sents_cot.append(sent)
            sent = ' {} {} '.format(src, bos_tok)
            edited_sents_only.append(sent)
            sent = ' {} {} '.format(src, bos_tok) + cot + ' {} '.format(eos_tok) + ans + ' {}'.format(eos_tok)
            edited_sents_all.append(sent)
            sent = ' {} {} '.format(src, bos_tok) + ans + ' {}'.format(eos_tok)
            edited_sents_nocot.append(sent)

        batch_encoding_cot = tokenizer(edited_sents_cot, add_special_tokens=True, truncation=True, max_length=self.max_len)
        batch_encoding_only = tokenizer(edited_sents_only, add_special_tokens=True, truncation=True, max_length=self.max_len)
        batch_encoding_all = tokenizer(edited_sents_all, add_special_tokens=True, truncation=True, max_length=self.max_len)
        batch_encoding_nocot = tokenizer(edited_sents_nocot, add_special_tokens=True, truncation=True, max_length=self.max_len)
        self.examples_cot = batch_encoding_cot["input_ids"]
        self.examples_only = batch_encoding_only["input_ids"]
        self.examples_all = batch_encoding_all["input_ids"]
        self.examples_nocot = batch_encoding_nocot["input_ids"]

        self.labels_cot = copy.deepcopy(self.examples_cot)
        self.labels_all = copy.deepcopy(self.examples_all)
        self.labels_cot_shift = copy.deepcopy(self.examples_cot)
        self.labels_nocot = copy.deepcopy(self.examples_nocot)

        self.src_sent_cot = []
        self.tgt_sent_cot = []

        temp_src_len = 0
        temp_tgt_len = 0
        temp_count = 0
        separator = tokenizer.eos_token_id #tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
        for i, elem in enumerate(self.labels_cot):
            sep_idx = elem.index(separator) + 1
            self.src_sent_cot.append(self.examples_cot[i][:sep_idx-1])
            self.tgt_sent_cot.append(self.examples_cot[i][sep_idx-1:])
            self.labels_cot[i][:sep_idx] = [-100] * sep_idx
            assert self.labels_all[i][sep_idx-1] == separator
            self.labels_all[i][:sep_idx] = [-100] * sep_idx
            self.labels_cot_shift[i][:sep_idx-1] = [-100] * (sep_idx-1)
            temp_src_len += sep_idx-1
            temp_tgt_len += len(elem) - (sep_idx-1)
            temp_count += 1

        print('tgt_avg: ', temp_tgt_len / temp_count)
        print('src_avg: ', temp_src_len / temp_count)
        print('ratios: ', temp_src_len/temp_tgt_len)

        self.src_sent_nocot = []
        self.tgt_sent_nocot = []
        temp_src_len = 0
        temp_tgt_len = 0
        temp_count = 0
        separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
        for i, elem in enumerate(self.labels_nocot):
            sep_idx = elem.index(separator) + 1
            self.src_sent_nocot.append(self.examples_nocot[i][:sep_idx-1])
            self.tgt_sent_nocot.append(self.examples_nocot[i][sep_idx-1:])
            self.labels_nocot[i][:sep_idx] = [-100] * sep_idx
            temp_src_len += sep_idx-1
            temp_tgt_len += len(elem) - (sep_idx-1)
            temp_count += 1

        print('tgt_avg: ', temp_tgt_len / temp_count)
        print('src_avg: ', temp_src_len / temp_count)
        print('ratios: ', temp_src_len/temp_tgt_len)

        print("---------------------------")
        print("Example features:")
        print(f"Full entry: {edited_sents_all[0]}")
        print(f"No CoT: {edited_sents_nocot[0]}")
        print(f"Only CoT: {edited_sents_cot[0]}")
        print(f"Product input: {edited_sents_only[0]}")
        #print(f"{self.labels_cot[0]}")
        #print(f"{self.labels_nocot[0]}")
        #print(self.examples_nocot[0])
        #print(self.src_sent_nocot[0])
        #print(self.tgt_sent_nocot[0])
        print("---------------------------")

    def __len__(self):
        return len(self.examples_cot)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return (torch.tensor(self.examples_cot[i], dtype=torch.long),
                torch.tensor(self.examples_nocot[i], dtype=torch.long),
                torch.tensor(self.labels_cot[i], dtype=torch.long),
                torch.tensor(self.labels_cot_shift[i], dtype=torch.long),
                torch.tensor(self.labels_nocot[i], dtype=torch.long),
                torch.tensor(self.src_sent_cot[i], dtype=torch.long),
                torch.tensor(self.src_sent_nocot[i], dtype=torch.long),
                torch.tensor(self.tgt_sent_cot[i], dtype=torch.long),
                torch.tensor(self.tgt_sent_nocot[i], dtype=torch.long),
                torch.tensor(self.examples_only[i], dtype=torch.long),
                torch.tensor(self.examples_all[i], dtype=torch.long),
                torch.tensor(self.labels_all[i], dtype=torch.long),
                )
    
    def readDataset(self):
        '''
        Automatically finds and reads existing datasets of the initialized @type and tokenizes it.
        '''
        self.__tokenizeDataset()

@dataclass
class CoTDataCollator:
    """
    VAEData collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        #import pdb; pdb.set_trace()
        input_ids_cot, input_ids_nocot, labels_cot, labels_cot_shift, labels_nocot, src_cot, src_nocot, tgt_cot, tgt_nocot, input_ids_only, input_ids_all, labels_all = zip(*examples)
        input_ids_cot = self._tensorize_batch(input_ids_cot)
        input_ids_cot[input_ids_cot.lt(0)] = self.tokenizer.eos_token_id
        input_ids_only = self._tensorize_batch(input_ids_only)
        input_ids_only[input_ids_only.lt(0)] = self.tokenizer.eos_token_id
        input_ids_all = self._tensorize_batch(input_ids_all)
        input_ids_all[input_ids_all.lt(0)] = self.tokenizer.eos_token_id
        input_ids_nocot = self._tensorize_batch(input_ids_nocot)
        input_ids_nocot[input_ids_nocot.lt(0)] = self.tokenizer.eos_token_id
        labels_cot = self._tensorize_batch(labels_cot)
        labels_all = self._tensorize_batch(labels_all)
        labels_cot_shift = self._tensorize_batch(labels_cot_shift)
        labels_nocot = self._tensorize_batch(labels_nocot)
        return {"input_ids_cot": input_ids_cot, "input_ids_nocot": input_ids_nocot, "labels_cot": labels_cot, "labels_cot_shift": labels_cot_shift, "labels_nocot": labels_nocot, 'input_ids_only': input_ids_only, 'input_ids_all': input_ids_all, 'labels_all': labels_all}

    def _tensorize_batch(self, examples):
        # In order to accept both lists of lists and lists of Tensors
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            return pad_sequence(examples, batch_first=True, padding_value=-100)