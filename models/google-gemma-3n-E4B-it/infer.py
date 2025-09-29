from typing import List, Optional, Union
import os
import logging
import torch
import requests
import base64
import soundfile as sf
from io import BytesIO
from urllib.request import urlopen
import re
from models import Message
from utils import cleanup_audio_files
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("infer")

MODEL = None
PROCESSOR = None
MODEL_PATH = os.environ.get("MODEL_PATH", "google/gemma-3n-E4B-it")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

def load_model():
    """Load the Gemma 3n model"""
    global MODEL, PROCESSOR

    if MODEL is None or PROCESSOR is None:
        try:
            logger.info(f"Loading model from {MODEL_PATH}")
            
            # Set Hugging Face token for accessing gated models
            os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
            
            # Load the model and processor
            MODEL = Gemma3nForConditionalGeneration.from_pretrained(
                MODEL_PATH, 
                device_map="auto",
                torch_dtype=torch.bfloat16
            ).eval()
            
            PROCESSOR = AutoProcessor.from_pretrained(MODEL_PATH)
            
            logger.info("Model and processor loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    return MODEL

def encode_audio_base64(audio_path: str) -> str:
    """Encode audio file to base64"""
    try:
        if audio_path.startswith("http"):
            # If it's a URL, download the content first
            response = requests.get(audio_path)
            response.raise_for_status()
            return base64.b64encode(response.content).decode("utf-8")
        else:
            # If it's a local file path
            with open(audio_path, "rb") as audio_file:
                return base64.b64encode(audio_file.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error encoding audio: {str(e)}")
        raise e

async def infer(
    model: str,
    messages: List[Message],
    temperature: float,
    max_tokens: int,
    audio_inputs: List[str],
    stop: Optional[Union[str, List[str]]] = None
) -> str:
    """
    Run inference with the Gemma 3n model
    """
    loaded_model = load_model()
    
    try:
        # Format messages for Gemma 3n following the sample code format
        gemma_messages = []
        
        for idx, message in enumerate(messages):
            # Create a message dictionary with role
            msg_dict = {"role": message.role}
            
            # Handle string content
            if isinstance(message.content, str):
                # For simple text messages, we'll create a list with a single text item
                msg_dict["content"] = [
                    {"type": "text", "text": message.content}
                ]
            # Handle list content (multimodal)
            elif isinstance(message.content, list):
                content_list = []
                
                # Process each content block
                for block in message.content:
                    if hasattr(block, 'type'):
                        if block.type == "text" and hasattr(block, 'text') and block.text:
                            content_list.append({
                                "type": "text",
                                "text": block.text
                            })
                        elif block.type == "audio_url" and hasattr(block, 'audio_url') and block.audio_url:
                            content_list.append({
                                "type": "audio",
                                "audio": block.audio_url.url
                            })
                        elif block.type == "input_audio" and hasattr(block, 'input_audio') and block.input_audio:
                            content_list.append({
                                "type": "audio",
                                "audio": block.input_audio.data
                            })
                
                # Add audio files to the first user message if we have any
                if message.role == "user" and idx == 0 and audio_inputs:
                    for audio_file in audio_inputs:
                        content_list.append({
                            "type": "audio",
                            "audio": audio_file
                        })
                
                msg_dict["content"] = content_list
            
            # Add the message to our list
            gemma_messages.append(msg_dict)
        
        # Use the processor's apply_chat_template method as shown in the sample
        try:
            model_inputs = PROCESSOR.apply_chat_template(
                gemma_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
        except Exception as e:
            logger.error(f"Error applying chat template: {str(e)}")
            logger.error(f"Messages structure: {gemma_messages}")
            raise
        
        # Move inputs to the same device as the model
        model_inputs = model_inputs.to(loaded_model.device, dtype=loaded_model.dtype)
        
        # Set generation parameters
        gen_kwargs = {
            "max_new_tokens": max_tokens if max_tokens else 512,
            "temperature": temperature if temperature else 1.0
        }
        
        if stop:
            gen_kwargs["stopping_criteria"] = stop
    
        # Generate response
        generation = loaded_model.generate(**model_inputs, **gen_kwargs)
        
        # Decode the response
        response = PROCESSOR.batch_decode(generation, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        
        # Clean up audio files
        cleanup_audio_files(audio_inputs)

        # Try to extract the assistant's response
        try:
            if "<assistant>" in response:
                assistant_response = response.split("<assistant>")[-1].strip()
            else:
                # If we can't find the assistant tag, return the whole response
                assistant_response = response.strip()
        except Exception as e:
            logger.error(f"Error extracting assistant response: {str(e)}")
            assistant_response = response.strip()
        
        return assistant_response
    except Exception as e:
        cleanup_audio_files(audio_inputs)
        logger.error(f"Error during inference: {str(e)}")
        raise e
