from pathlib import Path

class Config:
    # Main dataset path
    DATASET_ROOT = Path(r"C:\Users\aasri\OneDrive\Desktop\NLP\legal_doc_extraction\dataset")
    
    # Dataset structure paths
    PATHS = {
        'IN-Abs': {
            'test-data': {
                'judgement': DATASET_ROOT / "IN-Abs" / "test-data" / "judgement",
                'summary': DATASET_ROOT / "IN-Abs" / "test-data" / "summary"
            },
            'train-data': {
                'judgement': DATASET_ROOT / "IN-Abs" / "train-data" / "judgement",
                'summary': DATASET_ROOT / "IN-Abs" / "train-data" / "summary"
            }
        },
        'IN-Ext': {
            'judgement': DATASET_ROOT / "IN-Ext" / "judgement",
            'summary': {
                'full': DATASET_ROOT / "IN-Ext" / "summary" / "full",
                'segment-wise': DATASET_ROOT / "IN-Ext" / "summary" / "segment-wise"
            }
        }
    }

    @classmethod
    def verify_structure(cls):
        """Verify all expected paths exist"""
        missing = []
        for dataset in cls.PATHS.values():
            if 'test-data' in dataset:  # IN-Abs structure
                for split in ['test-data', 'train-data']:
                    for file_type in ['judgement', 'summary']:
                        path = dataset[split][file_type]
                        if not path.exists():
                            missing.append(str(path))
            else:  # IN-Ext structure
                if not dataset['judgement'].exists():
                    missing.append(str(dataset['judgement']))
                if not dataset['summary']['full'].exists():
                    missing.append(str(dataset['summary']['full']))
                if not dataset['summary']['segment-wise'].exists():
                    missing.append(str(dataset['summary']['segment-wise']))
        
        if missing:
            cls.print_structure()
            raise FileNotFoundError(
                f"Missing {len(missing)} paths:\n" + 
                "\n".join(f"• {p}" for p in missing)
            )
        return True

    @classmethod
    def print_structure(cls):
        """Display expected folder structure"""
        print("""
        Required Dataset Structure:
        dataset/
        ├── IN-Abs/
        │   ├── test-data/
        │   │   ├── judgement/
        │   │   └── summary/
        │   └── train-data/
        │       ├── judgement/
        │       └── summary/
        └── IN-Ext/
            ├── judgement/
            └── summary/
                ├── full/
                └── segment-wise/
        """)