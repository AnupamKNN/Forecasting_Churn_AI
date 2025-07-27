from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    regression_train_file_path: str
    regression_test_file_path: str
    classification_train_file_path: str
    classification_test_file_path: str