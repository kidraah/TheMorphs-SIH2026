import torch
from huggingface_hub import hf_hub_download
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class PreprocessingService:
    def __init__(self):
        self.sample_data = None

    def get_inference_tensors(self):
        """
        For the prototype, we download and use the provided holdout sample data
        from the Hugging Face repository to simulate real-time data ingestion.
        """
        if self.sample_data is None:
            logger.info("Downloading sample inference data...")
            sample_path = hf_hub_download(
                repo_id=settings.model_repo_id,
                filename="sample_inference_data.pt"
            )
            self.sample_data = torch.load(sample_path, map_location="cpu")
            logger.info("Sample inference data loaded.")
            
        imdaa = self.sample_data['imdaa']
        insat = self.sample_data['insat']
        terrain = self.sample_data['terrain']
        
        return imdaa, insat, terrain

preprocessing_service = PreprocessingService()
