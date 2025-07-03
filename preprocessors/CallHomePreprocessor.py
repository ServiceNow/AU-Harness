import logging
from tqdm import tqdm
logger = logging.getLogger(__name__)

class CallHomePreprocessor:
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
        total_duration = 0

        # Expecting dataset to be a dict-of-lists (same layout as AudiobenchPreprocessor)
        keys = list(dataset.keys())
        num_samples = len(dataset[keys[0]]) if keys else 0

        for i in tqdm(range(num_samples), desc="Preprocessing"):
            record_raw = {k: dataset[k][i] for k in keys}
            # Flatten audio/context
            array = record_raw["audio"]["array"]
            sampling_rate = record_raw["audio"]["sampling_rate"]

            # Initialize lists for chunked output
            all_chunks = []
            current_chunk = {"A": [], "B": []}

            s_speakers = [speaker[0] for speaker in record_raw['speakers']]
            s_starts = record_raw['timestamps_start']
            s_ends = record_raw['timestamps_end']
            #logger.info(f"s_speakers: {s_speakers}, s_starts: {s_starts}, s_ends: {s_ends}")
            for i in range(len(s_speakers)):
                speakerID = s_speakers[i]
                if speakerID not in current_chunk:
                    current_chunk[speakerID] = []
                # 30 second chunking for instructions
                _s_start_val = s_starts[i]
                _s_end_val = s_ends[i]
                adjusted_start_time = _s_start_val % 30
                adjusted_end_time = _s_end_val % 30
                #logger.info(f"SpeakerID: {speakerID}, adjusted_start_time: {adjusted_start_time}, adjusted_end_time: {adjusted_end_time}")
                if adjusted_end_time < adjusted_start_time:
                    #logger.info(f"current_chunk keys: {current_chunk.keys()}")
                    if sum(len(v) for v in current_chunk.values()) == 1:
                        # if B also overlaps, and chunk was already made
                        all_chunks[-1][speakerID].append([adjusted_start_time, 30])
                        current_chunk[speakerID].append([0, adjusted_end_time])
                        continue
                    # This segment crosses a 30-second boundary, triggering a new chunk.(normal)
                    current_chunk[speakerID].append([adjusted_start_time, 30])
                    #logger.info(f"current_chunk: {current_chunk}")
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
            #logger.info(f"all_chunks: {all_chunks}")
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
                instruction_text = base_instruct.replace("{speaker1}", str(speaker1)).replace("{audio1}", str(audio1))
                instruction_payload.append(instruction_text)
            reference = all_chunks
            record = {
                "array": array,
                "sampling_rate": sampling_rate,
                "instruction": instruction_payload,
                "reference": reference,
                "task_type": "SpeakerDiarization"
            }
            total_duration += len(record["array"]) / record["sampling_rate"]
            record["judge_type"] = properties.get("judge_type", "detailed")
            input_data.append(record)

        logger.info(f"Dataset is {total_duration / 3600:.2f} hours long")
        logger.info('\n=  =  =  Dataset Sample  =  =  =\n')
        logger.info(input_data[0].keys())
        logger.info(input_data[0])
        logger.info('\n=  =  =  =  =  =  =  =  =  =  =  =\n')

        return input_data
