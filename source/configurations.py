from transformers import PretrainedConfig
import os

class ThoughtEmulatorConfig(PretrainedConfig):
    def __init__(
        self,
        base_model='gpt2',
        tokenizer_name='gpt2',
        mixture_size=1,
        softmax_temperature=0.05,
        batch_size = 32,
        eta = 5e-5,
        max_grad_norm = 1,
        max_new_tokens = 17,
        epochs = 1,
        delta = 'dynamic', #Here we assign the dynamic delta paramter for variable length CoT
        subset = 'diagonal',
        **kwargs,
    ):
        self.base_model = base_model
        self.tokenizer_name = tokenizer_name
        self.mixture_size = mixture_size
        self.softmax_temperature = softmax_temperature
        self.batch_size = batch_size
        self.eta = eta
        self.max_grad_norm = max_grad_norm
        self.max_new_tokens = max_new_tokens
        self.epochs = epochs
        self.delta = delta
        self.subset = subset
        super().__init__(**kwargs)

class MindReadingEmulatorConfig(PretrainedConfig):
    def __init__(
        self,
        base_model='gpt2',
        tokenizer_name='gpt2',
        mixture_size=1,
        batch_size = 32,
        eta = 5e-5,
        max_grad_norm = 1,
        max_new_tokens = 13,
        epochs = 1,
        delta = 'dynamic', #Here we assign the dynamic delta paramter for variable length CoT
        subset = 'diagonal',
        **kwargs,
    ):
        self.base_model = base_model
        self.tokenizer_name = tokenizer_name
        self.mixture_size = mixture_size
        self.batch_size = batch_size
        self.eta = eta
        self.max_grad_norm = max_grad_norm
        self.max_new_tokens = max_new_tokens
        self.epochs = epochs
        self.delta = delta
        self.subset = subset
        super().__init__(**kwargs)

class TeacherConfig(PretrainedConfig):
    def __init__(
        self,
        base_model='gpt2',
        tokenizer_name='gpt2',
        batch_size = 32,
        eta = 5e-5,
        max_grad_norm = 1,
        max_new_tokens = 128,
        epochs = 1,
        **kwargs,
    ):
        self.base_model = base_model
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.eta = eta
        self.max_grad_norm = max_grad_norm
        self.max_new_tokens = max_new_tokens
        self.epochs = epochs
        super().__init__(**kwargs)

def save_model(model, tokenizer, model_dir):
    print ('saving', model_dir)
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
