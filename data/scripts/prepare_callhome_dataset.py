import os
import glob
import re
import logging
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm
import soundfile as sf
from datasets import Audio
from scipy.signal import resample

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def load_callhome_dataset(base_dir):
    """
    Load the CallHome English dataset from audio and transcript files.
    
    Args:
        base_dir: Base directory containing 'audio' and 'transcripts' folders
        
    Returns:
        pandas.DataFrame: DataFrame with columns 'id', 'audio', 'transcript'
    """
    audio_dir = os.path.join(base_dir, 'audio')
    transcript_dir = os.path.join(base_dir, 'transcripts')
    
    # Get all audio files
    audio_files = glob.glob(os.path.join(audio_dir, '*.wav'))
    audio_ids = [os.path.splitext(os.path.basename(f))[0] for f in audio_files]
    logging.info("Found %d audio files", len(audio_ids))
    
    # Get all transcript files
    transcript_files = glob.glob(os.path.join(transcript_dir, '*.cha'))
    transcript_ids = [os.path.splitext(os.path.basename(f))[0] for f in transcript_files]
    logging.info("Found %d transcript files", len(transcript_ids))
    
    # Find common IDs
    common_ids = sorted(set(audio_ids).intersection(set(transcript_ids)))

    logging.info("Found %d matching audio-transcript pairs", len(common_ids))
    
    # Create dataset
    data = []
    for file_id in tqdm(common_ids, "Processing audio-transcript pairs"):
        audio_path = os.path.join(audio_dir, f"{file_id}.wav")
        transcript_path = os.path.join(transcript_dir, f"{file_id}.cha")
        
        # Read audio file metadata
        try:
            audio_info = sf.info(audio_path)
            audio_duration = audio_info.duration
            sample_rate = audio_info.samplerate
            num_channels = audio_info.channels
        except Exception as e:
            logging.error("Error reading audio file %s: %s", audio_path, str(e))
            audio_duration = None
            sample_rate = None
            num_channels = None
            
        try:
            parse_result = extract_transcript_from_cha(Path(transcript_path))
            cleaned_lines, min_start_ms, max_end_ms = parse_result
        except Exception as e:
            logging.error("Error processing transcript file %s: %s", transcript_path, str(e))

        # audio_result = load_audio(audio_path)
            
        data.append({
            'id': file_id,
            'audio_path': audio_path,
            'audio_duration': audio_duration,
            'sample_rate': sample_rate,
            'num_channels': num_channels,
            'turns': cleaned_lines,
            'conversation_audio_start_time': min_start_ms,
            'conversation_audio_end_time': max_end_ms
        })
    
    df = pd.DataFrame(data)
    logging.info("Created dataset with %d samples", len(df))
    return df

def extract_transcript_from_cha(cha_path):
    """
    Parse the CHA file and clean up the text.
    
    Args:
        cha_path: Path to the CHA file
        
    Returns:
        Tuple of (cleaned_lines, min_start_ms, max_end_ms) or None if no valid lines
    """
    # Allow optional leading '*' and whitespace before the speaker code
    # We capture the speaker code and the entire remainder of the line (which may contain a \x15<start>_<end>\x15 timestamp)
    line_re = re.compile(r"^\s*\*?(?P<spkr>(?:[A-Z]+|PAR\d+)):\s*(?P<txt>.*)$")

    # Regex to capture timestamps of the form \x15<start>_<end>\x15 (numbers are in milliseconds)
    ts_re = re.compile(r"\x15(?P<start>[0-9]+)_(?P<end>[0-9]+)\x15")

    min_start_ms = None
    max_end_ms = None
    cleaned_lines = []  # List of tuples: (orig_spkr, cleaned_text, start_ms, end_ms)

    # First pass: clean up all text and collect lines with timestamps
    with cha_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = line_re.match(line)
            if not m:
                continue
            orig_spkr = m.group("spkr")
            raw_txt = m.group("txt")
            ts_match = ts_re.search(raw_txt)
            if not ts_match:
                continue  # Only process lines with timestamps
            start_ms = int(ts_match.group("start"))
            end_ms = int(ts_match.group("end"))
            if min_start_ms is None or start_ms < min_start_ms:
                min_start_ms = start_ms
            if max_end_ms is None or end_ms > max_end_ms:
                max_end_ms = end_ms
            text = raw_txt
            # Remove any word immediately following an = sign (e.g., =laughs, =lipsmack)
            text = re.sub(r'=\w+', '', text)
            text = re.sub(r'&\w+', '', text)
            text = ts_re.sub('', text)
            text = re.sub(r'\s+', ' ', text).strip()
            # Remove weird punctuation/annotation tokens
            text = re.sub(r'(?:[&+\-/]+[.,]*)+', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            # Remove if only punctuation or whitespace remains
            if not text or not re.search(r'\w', text) or all(c in r'.,;:!?-()[]{}"\'\'\s' for c in text):
                continue
            cleaned_lines.append((orig_spkr, text, start_ms, end_ms))

    if min_start_ms is None or max_end_ms is None or not cleaned_lines:
        return None

    return cleaned_lines, min_start_ms, max_end_ms

def to_mono(x):
    """
    Convert audio data to mono format.
    
    Args:
        x: Input audio data, can be 1D (already mono) or 2D (multi-channel)
        
    Returns:
        numpy.ndarray: Mono audio data as a 1D float32 array
        
    Raises:
        ValueError: If input audio has more than 2 dimensions
    """
    a = np.asarray(x)
    if a.ndim == 1:
        return a.astype(np.float32)
    if a.ndim == 2:  # channels x samples OR samples x channels
        # assume the smaller dim is channels (<=8)
        ax = 0 if a.shape[0] <= 8 else 1
        return a.mean(axis=ax).astype(np.float32)
    raise ValueError(f"Audio must be 1D or 2D, got shape {a.shape}")

def load_audio(audio_path):
    """
    Load audio data from file path
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        audio dictionary
    """
    try:
        audio_data, sample_rate = sf.read(audio_path)
        target_sr = 16000

        if sample_rate != target_sr:
            num_samples = int(round(audio_data.shape[0] * target_sr / sample_rate))
            audio_data = resample(audio_data, num_samples)
            sample_rate = target_sr
        return {"array": to_mono(audio_data), "sampling_rate": sample_rate}
    except Exception as e:
        logging.error("Error loading audio file %s: %s", audio_path, str(e))
        return None

def save_as_hf_dataset(df, output_dir=None):
    """
    Convert pandas DataFrame to HuggingFace Dataset format and save to disk
    
    Args:
        df: pandas DataFrame with the dataset
        output_dir: directory to save the dataset to
        
    Returns:
        DatasetDict: HuggingFace dataset
    """
    logging.info("Saving dataset to %s", output_dir)
    
    # Create a copy of the dataframe to avoid modifying the original
    df_save = df.copy()
    
    # Convert the 'turns' column to a serializable format
    # The error occurs because 'turns' contains tuples with mixed types
    # Convert to a list of dictionaries which is more compatible with Parquet
    if 'turns' in df_save.columns:
        df_save['turns'] = df_save['turns'].apply(lambda turns: [
            {
                'speaker': turn[0],
                'text': turn[1],
                'start_ms': int(turn[2]),
                'end_ms': int(turn[3])
            } if turns is not None else {}
            for turn in (turns or [])
        ])
    
    # Convert pandas DataFrame to HuggingFace Dataset first
    from datasets import Dataset
    hf_dataset = Dataset.from_pandas(df_save)
    
    # Now we can use cast_column on the HuggingFace Dataset
    # hf_dataset = hf_dataset.cast_column("audio", Audio())

    try:
        # Save the HuggingFace dataset as a Parquet file
        parquet_path = os.path.join(output_dir, "dataset.parquet")
        hf_dataset.to_parquet(parquet_path)
        logging.info("Successfully saved dataset to Parquet format")
    except Exception as e:
        logging.error("Error saving to HuggingFace format: %s", str(e))
        # Fallback to CSV if HuggingFace format fails
        csv_path = os.path.join(output_dir, 'dataset.csv')
        logging.info("Attempting to save as CSV to %s", csv_path)
        df_save.to_csv(csv_path, index=False)
        logging.info("Successfully saved dataset to CSV format")
    
    return hf_dataset

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create HuggingFace dataset from CallHome audio and transcript files')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Directory containing audio and transcripts folders')
    args = parser.parse_args()
    
    # Load dataset from specified directory
    df = load_callhome_dataset(args.data_dir)
    logging.info("Loaded %d samples", len(df))
    print(df.head())

    # Create hf_dataset directory in the same path as data_dir
    output_dir = os.path.join(args.data_dir, 'hf_dataset')
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to HuggingFace Dataset and save to disk
    save_as_hf_dataset(df, output_dir=output_dir)