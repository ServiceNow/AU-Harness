"""UTMOSv2 metric for TTS audio quality evaluation."""

import logging
import torch
import utmosv2
from metrics.metrics import Metrics
from utils.custom_logging import write_record_log, append_final_score
from utils import util
import tempfile
import shutil
import os
import warnings
from tqdm import tqdm

logging.getLogger("timm").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning, module="utmosv2")
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*CUDA is not available.*")

logger = logging.getLogger(__name__)


class UTMOSMetric(Metrics):
    """UTMOSv2 metric for evaluating TTS audio quality."""

    def __init__(self, batch_size=1):
        """Initialize UTMOSv2 metric.

        Args:
            batch_size: Number of audio files to process in parallel (default: 1)
        """
        super().__init__()
        self.name = "utmos"
        self.record_level_scores = None
        self.batch_size = batch_size

        # Load model once and reuse for all evaluations
        logger.info("[UTMOSMetric] Loading UTMOSv2 model...")
        self.model = utmosv2.create_model(pretrained=True)

        # Determine device
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        logger.info(f"[UTMOSMetric] Model loaded on {self.device} with batch_size={batch_size}")

    def __call__(self, candidates, references, instructions=None, *,
                 task_name: str | None = None, model_name: str | None = None,
                 model_responses=None):
        """Compute UTMOSv2 scores for TTS-generated audio.

        Args:
            candidates: List of audio file paths (from TTS generation)
            references: List of ground truth text (not used for UTMOS)
            instructions: Optional instructions
            task_name: Name of the task
            model_name: Name of the model
            model_responses: Raw model responses

        Returns:
            Dictionary with overall UTMOS score
        """
        self.instructions = instructions

        # Compute UTMOS scores for each audio file
        self.record_level_scores = self.compute_record_level_scores(candidates)

        # Calculate mean UTMOS
        scores = self.record_level_scores.get(self.name, [])
        valid_scores = [score for score in scores if score is not None]
        mean_utmos = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        overall_score = {self.name: util.smart_round(mean_utmos, 3)}

        if task_name and model_name:
            write_record_log(self, references, candidates, scores,
                           task_name, model_name, instructions=self.instructions)
            append_final_score(self, overall_score, task_name, model_name)

        return overall_score

    def compute_record_level_scores(self, audio_files: list) -> dict[str, list]:
        """Compute UTMOSv2 scores for each audio file in batches.

        Args:
            audio_files: List of paths to generated audio files

        Returns:
            Dictionary with UTMOS scores for each audio file
        """
        num_batches = (len(audio_files) + self.batch_size - 1) // self.batch_size
        all_scores = []

        for batch_idx in tqdm(range(num_batches), desc="UTMOS", total=num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(audio_files))
            batch_files = audio_files[start_idx:end_idx]

            # Create temp directory for this batch
            with tempfile.TemporaryDirectory() as temp_dir:
                # Copy batch files to temp directory with indices
                for i, audio_file in enumerate(batch_files):
                    temp_name = f"audio_{i:06d}.wav"
                    temp_path = os.path.join(temp_dir, temp_name)
                    shutil.copy(audio_file, temp_path)

                # Batch prediction
                results = self.model.predict(
                    input_dir=temp_dir,
                    device=self.device,
                    batch_size=self.batch_size,
                    num_workers=0,
                    verbose=False
                )

                # Extract scores for this batch
                batch_scores = []
                for i in range(len(batch_files)):
                    temp_name = f"audio_{i:06d}.wav"
                    score = None
                    for result in results:
                        if result.get('file_path', '').endswith(temp_name):
                            score = result.get('predicted_mos')
                            break
                    batch_scores.append(score)

                all_scores.extend(batch_scores)

        return {self.name: all_scores}
