from typing import List, Optional, Union
import os
import logging
import torch
from models import Message
from utils import cleanup_audio_files
from huggingface_hub import snapshot_download

from kimia_infer.api.kimia import KimiAudio


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("infer")

MODEL = None
MODEL_PATH = os.environ.get("MODEL_PATH", "moonshotai/Kimi-Audio-7B-Instruct")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """Load the Kimi Audio model"""
    global MODEL

    if MODEL is None:
        try:
            logger.info(f"Loading model from {MODEL_PATH}")
            
            if os.path.exists(MODEL_PATH):
                # Using a local model path
                logger.info(f"Using local model from: {MODEL_PATH}")
                model_path = MODEL_PATH
            elif '/' in MODEL_PATH:
                # Download model from Hugging Face Hub
                logger.info(f"Downloading model from Hugging Face Hub: {MODEL_PATH}")
                model_path = snapshot_download(repo_id=MODEL_PATH)
            else:
                raise ValueError(f"Invalid model path: {MODEL_PATH}")
            
            MODEL = KimiAudio(model_path=model_path, load_detokenizer=True)
            MODEL = MODEL.to(DEVICE)
            logger.info("Base model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    return MODEL

def format_messages_for_model(messages: List[Message]) -> str:
    """Format messages into a prompt for the model"""
    formatted_prompt = ""

    
    for message in messages:
        role_prefix = {
            "system": "System: ",
            "user": "User: ",
            "assistant": "Assistant: "
        }.get(message.role, "")
        
        if isinstance(message.content, str):
            formatted_prompt += f"{role_prefix}{message.content}\n"
        elif isinstance(message.content, list):
            content_text = ""
            for block in message.content:
                if block.type == "text" and block.text:
                    content_text += block.text + " "
            
            if content_text:
                formatted_prompt += f"{role_prefix}{content_text.strip()}\n"
    
    formatted_prompt += "Assistant: "
    return formatted_prompt

async def infer(
    model: str,
    messages: List[Message],
    temperature: float,
    max_tokens: int,
    audio_inputs: List[str],
    stop: Optional[Union[str, List[str]]] = None
) -> str:
    """
    Run inference with the Audio Flamingo model
    """
    loaded_model = load_model()
    
    try:
        prompt = []
        
        for audio_file in audio_inputs:
            sound = Sound(audio_file)
            prompt.append(sound)
        
        text_prompt = format_messages_for_model(messages)
        prompt.append(text_prompt)
        
        generation_config = loaded_model.default_generation_config
        if temperature is not None:
            generation_config.temperature = temperature
        if max_tokens is not None:
            generation_config.max_new_tokens = max_tokens
        
        logger.info(f"Generating response with temperature={temperature}, max_tokens={max_tokens}")
        response = loaded_model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        cleanup_audio_files(audio_inputs)
        
        if isinstance(response, str):
            return response
        else:
            try:
                return response.text
            except AttributeError:
                return str(response)
    except Exception as e:
        cleanup_audio_files(audio_inputs)
        logger.error(f"Error during inference: {str(e)}")
        raise e
