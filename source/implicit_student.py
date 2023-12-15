import os
import sys
import math
import numpy
import warnings
warnings.filterwarnings('ignore')

#For safe imports
file_directory = os.getcwd()
parent_directory = os.path.dirname(file_directory)
sys.path.insert(False, parent_directory)

import torch
import numpy as np
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import tqdm
import logging
import inspect
from itertools import chain

from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, GenerationConfig, LogitsProcessorList
from torch.utils.data import DataLoader

from data.data import DatasetHandler
from data.data import CoTDataCollator
from data.data import extractAnswer

from source.configurations import MindReadingEmulatorConfig
from source.utils import get_sep_position, DoubleEOSStoppingCriteria, DoubleEOSLogitsProcessor, createAccuracyPlot, createLossPlot
from source.gpt2_implicit import GPT2LMHeadImplicitModel

from source.thought_emulator import ThoughtEmulator
from source.mindreading_emulator import MindReadingEmulator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logging.disable(logging.WARNING) # disable WARNING, INFO and DEBUG logging everywhere

class ImplicitStudent():
    def __init__(self, config : MindReadingEmulatorConfig, mindread : MindReadingEmulator, thought : ThoughtEmulator):
        super().__init__()
        self.mindread = mindread
        self.thought = thought
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    
    @torch.no_grad() #Freeze gradients.
    def evaluate(self, dataloader : DataLoader, ctx,):
        '''
        Calculates accuracy metrics on test data.
        '''
        self.thought.eval() #Freeze loss function, gradients etc.
        self.mindread.eval() 

        total_instances = 0
        total_tokens = 0
        total_correct = 0
        total_correct_tokens = 0
        total_loss = 0
        sub_iteration = 0
        
        for batch in tqdm.tqdm(dataloader):
            input_ids_nocot = batch['input_ids_nocot'].to(device)
            labels_nocot = batch['labels_nocot'].to(device)
            batch_size = input_ids_nocot.shape[0]
            with ctx:
                beam_output = self.mindread.generate(
                    input_ids=input_ids_nocot,
                    max_new_tokens=self.config.max_new_tokens,
                )

                beam_output = [inner_tensor.tolist()[0] for inner_tensor in beam_output]

                beam_output = torch.tensor(beam_output, dtype=torch.long).to(device)
                outputs = self.mindread.computeLoss(input_ids=beam_output, labels=labels_nocot)
                loss = outputs.loss
                token_accuracy = outputs.token_accuracy.item()

            total_loss += outputs.total_loss.item()
            total_correct_tokens += outputs.total_correct.item()
            total_tokens += outputs.total_tokens
            total_instances += batch_size

            # Generate
            with ctx:
                beam_output = self.mindread.generate(
                    input_ids=input_ids_nocot,
                    max_new_tokens=self.config.max_new_tokens,
                )

            # Evaluate
            sep_positions = get_sep_position(input_ids_nocot, self.tokenizer.eos_token_id)
            for i, (input_ids_i, beam_output_i) in enumerate(zip(input_ids_nocot, beam_output)):
                sub_iteration += 1
                sep_position = sep_positions[i].item()
                tgt = input_ids_i[sep_position+1:]
                tgt_text = self.tokenizer.decode(tgt, skip_special_tokens=True)
                ans = extractAnswer(tgt_text)
                pred_text = self.tokenizer.decode(beam_output_i[0][sep_position+1:], skip_special_tokens=True)
                pred_ans = extractAnswer(pred_text)
                if ans == pred_ans:
                    total_correct += 1
                if sub_iteration >= len(dataloader.dataset)-2: # to limit spam of prediction examples.
                    print (f'Input: {self.tokenizer.decode(input_ids_i[:sep_position], skip_special_tokens=True)}')
                    print (f'Target: {tgt_text}')
                    print (f'Predicted: {pred_text}')
                    print("")
        accuracy = total_correct / total_instances
        token_accuracy = total_correct_tokens / total_tokens
        loss = total_loss / total_tokens
        ppl = math.exp(loss)
        return accuracy, token_accuracy, ppl
    
    def predict(self, custom_data_handler : DatasetHandler) -> None:
        '''
        Used for custom test cases for fun. You can create custom test cases using the generateDataset using a DatasetHandler.
        '''

        dtype = 'float32'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)

        # Load data
        tokenizer = self.tokenizer
        collate_fn = CoTDataCollator(tokenizer)
        custom_data_handler = DataLoader(custom_data_handler, batch_size = self.config.batch_size, collate_fn = collate_fn, shuffle=False)

        self.evaluate(custom_data_handler, ctx)

    def trainModel(self, train_handler : DatasetHandler, test_handler : DatasetHandler, limit : float) -> None:
        '''
        Trains the model and automatically evaluates. 
        @limit hard caps the desired accuracy to stop training early if the threshold is met.
        '''
        dtype = 'float32'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)

        # Create Emulator 
        thought  = self.thought.to(device).to(ptdtype)
        mindread = self.mindread.to(device).to(ptdtype)

        # Load data
        tokenizer = self.tokenizer
        collate_fn = CoTDataCollator(tokenizer)
        train_dataloader = DataLoader(train_handler, batch_size = self.config.batch_size, collate_fn = collate_fn, shuffle=True)
        val_dataloader = DataLoader(test_handler, batch_size = self.config.batch_size, collate_fn = collate_fn, shuffle=False)

        trainable_params = chain(mindread.parameters(), thought.parameters())
        use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(trainable_params, lr=self.config.eta, **extra_args)

        thought.eval() # to turn off dropout
        mindread.eval() # to turn off dropout

        train_losses = []

        train_accs = []

        # Train
        iteration = 0
        for batch in tqdm.tqdm(train_dataloader):
            input_ids_nocot = batch['input_ids_nocot'].to(device)
            labels_nocot = batch['labels_nocot'].to(device)
            #print(labels_nocot)
            with ctx:
                with torch.no_grad():
                    

                    beam_output = self.mindread.generate(
                        input_ids=input_ids_nocot,
                        max_new_tokens=self.config.max_new_tokens,
                    )

                    beam_output = [inner_tensor.tolist()[0] for inner_tensor in beam_output]

                beam_output = torch.tensor(beam_output, dtype=torch.long).to(device)
                outputs = self.mindread.computeLoss(input_ids=beam_output, labels=labels_nocot)
            loss = outputs.loss
            token_accuracy = outputs.token_accuracy.item()

            #Stop training early to save resources.
            if token_accuracy > limit:
                print(f"Training accuracy limit reached, stopping training at training accuracy: {token_accuracy:.6f}.")
                break

            loss.backward() #Calculates gradients
            torch.nn.utils.clip_grad_norm_(trainable_params, self.config.max_grad_norm)
            optimizer.step() #Subtracts gradients.
            optimizer.zero_grad() #Set gradients to zero.
            ppl = loss.exp().item()

            #We want 10 updates on steps, accuracy and loss.
            if iteration % math.floor(len(train_dataloader)/10+1) == 0:
                print (f"Step: {iteration}. CrossEntropyLoss: {loss:.6f}. Training Accuracy: {token_accuracy:.6f}.")
            iteration += 1

            train_losses.append(loss.item())
            train_accs.append(token_accuracy)

        print (f"\u2714 Evaluating test dataset now...")
        accuracy, token_accuracy, ppl = self.evaluate(val_dataloader, ctx)

        print (f'\u2192 Perplexitity: {ppl:.6f}; Test Accuracy: {accuracy:.6f}; Training Accuracy: {token_accuracy:.6f}.')
        self.thought.save_pretrained(os.path.join(train_handler.path+r'\models\implict_thought_emulator'))
        self.mindread.save_pretrained(os.path.join(train_handler.path+r'\models\implict_mindreading_emulator'))

        createLossPlot(train_losses) #Plots the loss and accuracy information over batches, so we can gage training performance.
        createAccuracyPlot(train_accs)