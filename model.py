from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
import torch
import gc

model, tokenizer, image_processor, context_len = None, None, None, None
model_loaded = False

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
    
def load_model(model_path, arg=None):
    global model, tokenizer, image_processor, context_len, model_loaded
    print(f"Loading LLaVa model...It takes time.")
    if arg is not None:
        # arg is a str. arg_8bit and arg_4bit is bool.
        arg_8bit = "8bit" in arg
        arg_4bit = "4bit" in arg
    else:
        arg_8bit = False
        arg_4bit = False
    if arg_4bit:
        print("Loading 4-bit quantized model.")
    elif arg_8bit:
        print("Loading 8-bit quantized model.")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        load_8bit=arg_8bit,
        load_4bit=arg_4bit
    )
    #model.to(get_device())  # Move the model to the appropriate device
    print("Finish loading LLaVa model!")
    if model is not None:
        print("Model loaded successfully")
        print(f'Model loaded at id: {id(model)}')
        model_loaded = True
    else:
        print("Failed to load the model")

def unload_model():
    global model, tokenizer, image_processor, context_len, model_loaded
    if model_loaded:
        print("Unloading model")
        del model
        del tokenizer
        del image_processor
        del context_len
        gc.collect()
        torch.cuda.empty_cache()
        model, tokenizer, image_processor, context_len = None, None, None, None
        model_loaded = False
        print("Model unloaded successfully")
    else:
        print("Model is not loaded")
