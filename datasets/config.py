"""
Configuration for ViT-Lips dataset loading
"""

import os

class DatasetConfig:
    """Configuration class for dataset parameters"""
    
    # Dataset paths (modify these for your setup)
    DATA_DIR = "/path/to/your/lrs2_or_lrs3_data"
    
    # Dataset splits
    TRAIN_SPLIT = "train"
    VALID_SPLIT = "valid"
    TEST_SPLIT = "test"
    
    # Label types
    TEXT_LABELS = "txt"      # For sentence-level text
    PHONEME_LABELS = "phn"   # For phoneme-level labels
    
    # Video processing
    IMAGE_SIZE = 224         # Resize frames to this size
    MAX_FRAMES = 500         # Maximum frames per video
    MIN_FRAMES = 10          # Minimum frames per video
    
    # File extensions
    MANIFEST_EXT = ".tsv"
    DICT_FILE = "dict.phn.txt"
    
    @classmethod
    def get_manifest_path(cls, data_dir, split):
        """Get path to manifest file"""
        return os.path.join(data_dir, f"{split}{cls.MANIFEST_EXT}")
    
    @classmethod
    def get_label_path(cls, data_dir, split, label_type):
        """Get path to label file"""
        return os.path.join(data_dir, f"{split}.{label_type}")
    
    @classmethod
    def get_dict_path(cls, data_dir):
        """Get path to phoneme dictionary"""
        return os.path.join(data_dir, cls.DICT_FILE)
    
    @classmethod
    def validate_data_dir(cls, data_dir, split="valid", label_type="txt"):
        """
        Validate that data directory has required files
        
        Returns:
            valid: Boolean indicating if directory is valid
            missing: List of missing files
        """
        missing = []
        
        # Check manifest file
        manifest_path = cls.get_manifest_path(data_dir, split)
        if not os.path.exists(manifest_path):
            missing.append(manifest_path)
        
        # Check label file
        label_path = cls.get_label_path(data_dir, split, label_type)
        if not os.path.exists(label_path):
            missing.append(label_path)
        
        # Check phoneme dictionary if using phonemes
        if label_type == "phn":
            dict_path = cls.get_dict_path(data_dir)
            if not os.path.exists(dict_path):
                missing.append(dict_path)
        
        return len(missing) == 0, missing


# Example configurations for common datasets
class LRS2Config(DatasetConfig):
    """Configuration for LRS2 dataset"""
    pass


class LRS3Config(DatasetConfig):
    """Configuration for LRS3 dataset"""
    pass
