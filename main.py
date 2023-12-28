import csv
from io import BytesIO
import json
import os
from pathlib import Path
import requests
from PIL import Image
import torch
import gradio as gr
from tqdm import tqdm
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from model import context_len, load_model, unload_model
from run_llava import eval_model

# Set the model-related global variables
# 保存模型路径的文件
SAVED_MODEL_PATH_FILE = "model_path.txt"

# 读取保存的模型路径
def read_saved_model_path():
    with open('model_path.txt', 'r') as file:
        lines = file.readlines()
        directory = lines[0].strip()  # Read the directory and remove any trailing newlines/spaces
        # Check if there is a second line with parameters
        parameter = lines[1].strip() if len(lines) > 1 else None

    return directory, parameter

model_path, parameter = read_saved_model_path()

load_model(model_path, parameter)

# Function definitions remain the same
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def gen_caption(image, prompt, temperature=1.0, top_p=None, num_beams=1):
    # Convert the image into a URL or a file path
    if isinstance(image, Image.Image):
        # Save the image to a temporary file if it's a PIL image
        temp_image_file = "temp_image.png"
        image.save(temp_image_file)
        image_path = temp_image_file
    else:
        # Otherwise, assume the input is a file path or a URL
        image_path = image

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_path,
        "sep": os.sep,
        "temperature": temperature,
        "top_p": top_p,
        "num_beams": num_beams,
        "max_new_tokens": context_len  # Assuming context_len is defined globally
    })()

    result = eval_model(args)
    if isinstance(image, Image.Image):
        # Clean up the temporary image file
        os.remove(temp_image_file)
    return result


def save_csv_f(caption, output_dir, image_filename):
    type = 'a' if os.path.exists(f'{output_dir}/blip2_caption.csv') else 'x'
    with open(f'{output_dir}/blip2_caption.csv', type, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        csvlist = [image_filename]
        csvlist.extend(caption.splitlines())
        writer.writerow(csvlist)


def save_txt_f(caption, output_dir, image_filename):
    if os.path.exists(f'{output_dir}/{os.path.splitext(image_filename)[0]}.txt'):
        f = open(f'{output_dir}/{os.path.splitext(image_filename)[0]}.txt', 'w', encoding='utf-8')
    else:
        f = open(f'{output_dir}/{os.path.splitext(image_filename)[0]}.txt', 'x', encoding='utf-8')
    f.write(f'{caption}\n')
    f.close()
        
        

def prepare(image, process_type, input_dir, output_dir, save_csv, save_txt, prompt, temperature, top_p, num_beams):
    if process_type == "Single Image":
        return gen_caption(image, prompt, temperature, top_p, num_beams)
    elif process_type == "Batch Process":
        # Validate directories
        input_dir_path = Path(input_dir).resolve()
        output_dir_path = Path(output_dir).resolve()
        if not input_dir_path.is_dir():
            return "Input directory does not exist"
        if not output_dir_path.is_dir():
            output_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Get a list of images
        image_files = [
            f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        # Initialize tqdm progress bar
        progress_bar = tqdm(total=len(image_files), desc='Processing Images', unit='img')

        for image_filename in image_files:
            image = Image.open(f"{input_dir}/{image_filename}")
            print(f"Processing {image_filename}")
            caption = gen_caption(image, prompt, temperature, top_p, num_beams)
            if save_csv:
                save_csv_f(caption, output_dir, image_filename)
            if save_txt:
                save_txt_f(caption, output_dir, image_filename)
            image.close()
            
            # Update the progress bar
            progress_bar.update(1)

        # Close the progress bar
        progress_bar.close()
        
        return f"Processed {len(image_files)} images!"
    
                

# Define a separate function for single image processing
def prepare_single_image(input_image, temperature, top_p, num_beams, prompt):
    return prepare(input_image, "Single Image", None, None, False, False,
                   prompt, temperature, top_p, num_beams)

# and another for batch processing
def prepare_batch(input_dir, output_dir, save_csv, save_txt, prompt, temperature, top_p, num_beams):
    return prepare(None, "Batch Process", input_dir, output_dir, save_csv, save_txt,
                   prompt, temperature, top_p, num_beams)

# Main Gradio interface
def gui():
    with gr.Blocks() as demo:
        gr.Markdown("# LLaVa Caption Generator")

        with gr.Tabs() as tabs:
            with gr.TabItem("Single Image"):
                input_image = gr.Image(label="Image", type='pil')
                output_text_single = gr.Textbox(label="Generated Caption(s)", lines=10, placeholder="Generated captions will appear here...")
                temperature = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, label="Temperature")
                top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, label="Top p")
                num_beams = gr.Slider(minimum=1, maximum=10, value=1, label="Number of Beams", step=1)
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your description prompt here.")
                generate_caption_btn = gr.Button("Generate Caption", variant="primary")
                generate_caption_btn.click(prepare_single_image, inputs=[input_image, temperature, top_p, num_beams, prompt], outputs=output_text_single)

            with gr.TabItem("Batch Process"):
                input_dir = gr.Textbox(label="Input Directory", placeholder="Enter the directory path...")
                output_dir = gr.Textbox(label="Output Directory", placeholder="Enter the output directory path...")
                save_csv = gr.Checkbox(label="Save as CSV", value=True)
                save_txt = gr.Checkbox(label="Save as TXT", value=False)
                temperature = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, label="Temperature")
                top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, label="Top p")
                num_beams = gr.Slider(minimum=1, maximum=10, value=1, label="Number of Beams", step=1)
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your description prompt here.")
                output_text_batch = gr.Textbox(label="Batch Process Status", lines=10, placeholder="Batch processing status will appear here...")
                batch_process_btn = gr.Button("Process Batch", variant="primary")
                batch_process_btn.click(prepare_batch, inputs=[input_dir, output_dir, save_csv, save_txt, prompt, temperature, top_p, num_beams], outputs=output_text_batch)

        # Removed the Accordion block because there's no longer a switch between beam search and nucleus sampling
    return demo

if __name__ == "__main__":
    # Run the GUI
    app = gui()
    app.launch()