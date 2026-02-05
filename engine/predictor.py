import torch
import torch.nn.functional as F
import os
import yaml
from PIL import Image
from models.model_factory import build_model
from data.transforms import get_val_transforms

class Predictor:
    def __init__(self, config_path, checkpoint_path, use_cpu=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() and not use_cpu else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load Config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Build Model
        print("Building model...")
        self.model = build_model(self.config)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load Checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            except Exception:
                import numpy
                torch.serialization.add_safe_globals([
                    numpy.core.multiarray.scalar, 
                    numpy.dtype,
                    numpy.float64,
                    numpy.int64
                ])
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                
            state_dict = checkpoint['model_state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            self.model.load_state_dict(new_state_dict)
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found.")

        # Transforms
        img_size = self.config['data']['img_size']
        if isinstance(img_size, list):
            img_size = tuple(img_size)
        self.transform = get_val_transforms(img_size)

    def preprocess(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img)
        return img_tensor.unsqueeze(0).to(self.device)

    def predict(self, img1_path, img2_path):
        try:
            t1 = self.preprocess(img1_path)
            t2 = self.preprocess(img2_path)
        except Exception as e:
            print(f"Error processing images: {e}")
            return None

        with torch.no_grad():
            emb1 = self.model(t1)
            emb2 = self.model(t2)
            
            emb1 = F.normalize(emb1, p=2, dim=1)
            emb2 = F.normalize(emb2, p=2, dim=1)
            
            similarity = torch.mm(emb1, emb2.t()).item()
            
        return similarity
