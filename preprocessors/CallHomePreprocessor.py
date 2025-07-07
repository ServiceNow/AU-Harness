import logging
logger = logging.getLogger(__name__)

import random
import requests
from bs4 import BeautifulSoup
import os

class CallHomePreprocessor:
    def download_media(self, dataset, properties=None):
        """
        Downloads the CallHome dataset or required files for preprocessing.
        Args:
            dataset: str or dict, information about the dataset or its location.
            properties: dict, optional, additional download properties (e.g., destination path).
        Returns:
            list: List of paths to the downloaded files.
        """

        if properties is None:
            properties = {}

        # Example: set default download URL and destination
        username = properties.get('username', 'default_username')
        password = properties.get('password', 'default_password')
        base_url = properties.get('base_url', 'https://example.com/callhome-dataset/')
        # Always use private_datasets for downloaded media
        dest_dir = properties.get('dest_dir', './private_datasets/callhome_media')
        os.makedirs(dest_dir, exist_ok=True)

        auth = (username, password)
        response = requests.get(base_url, auth=auth)
        soup = BeautifulSoup(response.text, 'html.parser')

        downloaded_files = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if href.endswith('.mp3'):
                file_url = base_url + href
                file_path = os.path.join(dest_dir, href)
                if os.path.exists(file_path):
                    logger.info(f"File already exists at {file_path}. Skipping download.")
                    downloaded_files.append(file_path)
                    continue
                try:
                    logger.info(f"Downloading {file_url} to {file_path}...")
                    response = requests.get(file_url, auth=auth, stream=True)
                    response.raise_for_status()
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    logger.info(f"Download completed: {file_path}")
                    downloaded_files.append(file_path)
                except Exception as e:
                    logger.error(f"Failed to download file: {e}")
                    raise

        return downloaded_files

    def process(self, dataset, properties=None):
        """
        Processes a dataset for CallHome speaker diarization.
        Args:
            dataset: list of dicts, each with keys 'audio', 'timestamps_start', 'timestamps_end', 'speakers' (the last three are lists)
            properties: dict, optional
        Returns:
            input_data: list of processed dicts
        """
        if properties is None:
            properties = {}
        input_data = []
        for sample in dataset:
            audio = sample['audio']
            # Initialize lists for chunked output
            all_chunks = []
            current_chunk = {"A": [], "B": []}

            s_speakers = [speaker[0] for speaker in sample['speakers']]  # Extract first letter of each speaker ID
            s_starts = sample['timestamps_start']
            s_ends = sample['timestamps_end']

            for i in range(len(s_speakers)):
                speakerID = s_speakers[i]
                if speakerID not in current_chunk:
                    current_chunk[speakerID] = []
                # 30 second chunking for instructions
                _s_start_val = s_starts[i]
                _s_end_val = s_ends[i]
                adjusted_start_time = _s_start_val % 30
                adjusted_end_time = _s_end_val % 30

                if adjusted_end_time < adjusted_start_time:
                    if len(current_chunk["A"]) + len(current_chunk["B"]) == 1:
                        # if B also overlaps, and chunk was already made
                        all_chunks[-1][speakerID].append([adjusted_start_time, 30])
                        current_chunk[speakerID].append([0, adjusted_end_time])
                        continue
                    # This segment crosses a 30-second boundary, triggering a new chunk.(normal)
                    current_chunk[speakerID].append([adjusted_start_time, 30])
                    all_chunks.append(current_chunk)
                    current_chunk = {speakerID: [[0, adjusted_end_time]]}
                else:
                    # Segment does not cross a 30-second boundary in a way that triggers a new chunk.
                    # It's added to the current ongoing chunk.
                    current_chunk[speakerID].append([adjusted_start_time, adjusted_end_time])

            # After processing all segments for the sample, if there's a pending current_chunk, add it.
            # This handles cases where the last segment(s) did not trigger the 'if' condition.
            if current_chunk:
                all_chunks.append(current_chunk)

            instruction_payload = []
            base_instruct = """
                You are given a 30-second audio clip featuring two speakers. Your task is to perform speaker diarization on the clip.
                There may be speaker overlaps, and a speaker may have consecutive turns.
                For each segment where a speaker is talking, identify:
                - The speaker ("A" or "B")
                - The start timestamp (in seconds, relative to the start of the clip)
                - The end timestamp (in seconds, relative to the start of the clip)

                The output should be a JSON object. Each entry should be an object with the speaker as the key and a list `[start_time, end_time]` as the value. Multiple entries per speaker are allowed. Overlaps in speaker segments may occur, and a speaker may have consecutive turns.

                Example format:
                ```json
                {
                "A": [[0, 1.7], [3.4, 3.7],... [3.5, 5.4]],
                "B": [[1.6, 3.3], [3.9, 8.6],... [29.3, 30]]
                }
                ```

                The first speaker is {speaker1}, and the first audio clip is on {audio1} in seconds.
                So, start with 
                ```json
                {speaker1}: {audio1}
                ```
                """
            for chunk in all_chunks:
                speaker1 = list(chunk.keys())[0]
                audio1 = chunk[speaker1][0] if chunk[speaker1] else None
                instruction_payload.append(base_instruct.format(speaker1=speaker1, audio1=audio1))
            reference = all_chunks
            input_data.append({
                "audio": audio,
                "instruction": instruction_payload,  # Use the newly created structure
                "reference": reference,
                "task_type": "SpeakerDiarization"
            })

        logging.info('\n=  =  =  Dataset Sample  =  =  =')
        if input_data:
            logging.info(random.sample(input_data, 1)[0])
        logging.info('=  =  =  =  =  =  =  =  =  =  =  =\n')

        return input_data
