import os
import torch
from huggingface_hub import hf_hub_download
import logging
from app.config import settings
from app.services.varuna_model import SpatiotemporalMultiTaskModel

logger = logging.getLogger(__name__)

class ModelService:
    _instance = None

    def __init__(self):
        if ModelService._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            ModelService._instance = self
            self.model = None
            self.device = torch.device('cuda' if torch.cuda.is_available() and settings.use_gpu else 'cpu')
            self._initialize_model()

    def _initialize_model(self):
        logger.info(f"Loading VARUNA model onto {self.device}...")
        try:
            # Download/Cache weights from Hugging Face
            ckpt_path = hf_hub_download(
                repo_id=settings.model_repo_id, 
                filename=settings.model_filename
            )
            
            # Initialize architecture
            self.model = SpatiotemporalMultiTaskModel()
            
            # Load weights
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def predict(self, imdaa_tensor: torch.Tensor, insat_tensor: torch.Tensor, terrain_tensor: torch.Tensor):
        """
        Runs the forward pass through the VARUNA model.
        Returns probabilities (not logits).
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
            
        imdaa = imdaa_tensor.to(self.device)
        insat = insat_tensor.to(self.device)
        terrain = terrain_tensor.to(self.device)
        
        with torch.no_grad():
            logits = self.model(imdaa, insat, terrain)
            probs = torch.sigmoid(logits)
            
        # Move back to CPU for subsequent processing
        return probs.cpu()

def get_model_service() -> ModelService:
    if ModelService._instance is None:
        ModelService()
    return ModelService._instance
