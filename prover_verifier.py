from typing import Tuple, List, Dict
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Enable detailed error tracing
os.environ['RUST_BACKTRACE'] = 'full'

class DNASequenceProcessor:
    """Handles DNA sequence data loading and preprocessing."""
    
    DNA_MAPPING = {
        'A': [1,0,0,0],
        'C': [0,1,0,0],
        'G': [0,0,1,0],
        'T': [0,0,0,1]
    }
    
    @staticmethod
    def load_from_url(url: str) -> Tuple[np.ndarray, np.ndarray, LabelEncoder, np.ndarray]:
        """
        Load and preprocess DNA sequence data from UCI repository.
        
        Args:
            url: URL to the DNA sequence dataset
            
        Returns:
            Tuple containing processed features, one-hot encoded labels,
            label encoder, and encoded labels
        """
        response = requests.get(url)
        if response.status_code != 200:
            raise ConnectionError(f"Failed to download data: HTTP {response.status_code}")
        
        # Parse data
        data = [
            [parts[0], ''.join(parts[1:]).upper()]
            for line in response.text.split('\n')
            if (parts := line.strip().split()) and len(parts) >= 2
        ]
        
        df = pd.DataFrame(data, columns=['class', 'sequence'])
        
        return DNASequenceProcessor._process_sequences(df)
    
    @classmethod
    def _process_sequences(cls, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, LabelEncoder, np.ndarray]:
        """Process DNA sequences into numerical format."""
        # Encode sequences
        X_encoded = np.array([
            [cls.DNA_MAPPING[base] for base in seq if base in cls.DNA_MAPPING]
            for seq in df['sequence'].values
        ])
        
        # Pad sequences to same length
        max_length = max(len(seq) for seq in X_encoded)
        X_encoded = np.array([
            np.pad(seq, ((0, max_length - len(seq)), (0, 0)), 'constant')
            for seq in X_encoded
        ])
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(df['class'].values)
        y_onehot = to_categorical(y_encoded)
        
        return X_encoded, y_onehot, label_encoder, y_encoded

    @staticmethod
    def prepare_for_model(data: np.ndarray) -> List[float]:
        """
        Process DNA sequence data for model input.
        
        Args:
            data: Input sequence data
            
        Returns:
            Flattened list of processed data
        """
        data = np.array(data, dtype=np.float32)
        
        # Add batch dimension if needed
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)
        
        expected_shape = (57, 4)
        if data.shape[1:] != expected_shape:
            raise ValueError(f"Expected shape (batch_size, 57, 4), got {data.shape}")
        
        return data.flatten().tolist()

class ProofManager:
    """Handles proof generation and verification operations."""
    
    def __init__(self, cli_path: str):
        self.cli_path = Path(cli_path)
        if not self.cli_path.exists():
            raise FileNotFoundError(f"CLI tool not found at: {cli_path}")
    
    def save_input(self, data: List[float], output_path: str = "input.json") -> str:
        """Save processed data to JSON."""
        with open(output_path, "w") as f:
            json.dump([data], f, indent=4)
        return output_path
    
    def generate_proof(self, onnx_path: str, input_path: str, 
                      proof_path: str = "proof.json") -> bool:
        """Generate proof using mina-zkml-cli."""
        cmd = [
            str(self.cli_path), "proof",
            "-m", onnx_path,
            "-i", input_path,
            "-o", proof_path,
            "--input-visibility", "public",
            "--output-visibility", "public"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(f"Command output:\n{result.stdout}")
        
        if result.returncode != 0:
            print(f"Error generating proof:\n{result.stderr}")
            return False
            
        print(f"Proof generated at {proof_path}")
        return True
    
    def verify_proof(self, proof_path: str, input_path: str, 
                    output_path: str, onnx_path: str) -> bool:
        """Verify generated proof."""
        # Extract and save output data
        try:
            with open(proof_path, "r") as f:
                proof_data = json.load(f)
            
            if "output" in proof_data:
                with open(output_path, "w") as f:
                    json.dump(proof_data["output"], f, indent=4)
            else:
                print("No 'output' field in proof.json")
                return False
                
        except Exception as e:
            print(f"Error processing proof data: {e}")
            return False
        
        # Verify proof
        cmd = [
            str(self.cli_path), "verify",
            "-m", onnx_path,
            "-i", input_path,
            "-p", proof_path,
            "-o", output_path,
            "--input-visibility", "public",
            "--output-visibility", "public"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"Proof verification failed:\n{result.stderr}")
            return False
            
        return True

def main():
    # Configuration
    CLI_PATH = "/Users/eric/Code/Python/mina-zkml/examples/notebooks/mina-zkml-cli"
    ONNX_PATH = "/Users/eric/Code/Python/mina-zkml/examples/notebooks/2_dna_sequence_classifier.onnx"
    DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data"
    
    try:
        # Initialize processors
        proof_manager = ProofManager(CLI_PATH)
        
        # Load and prepare data
        X_encoded, y_onehot, _, _ = DNASequenceProcessor.load_from_url(DATA_URL)
        X_train, X_test, _, _ = train_test_split(
            X_encoded, y_onehot, test_size=0.2, random_state=42
        )
        
        # Process test sample
        test_sample = X_test[0]
        processed_data = DNASequenceProcessor.prepare_for_model(test_sample)
        
        # Generate and verify proof
        input_path = proof_manager.save_input(processed_data)
        
        if proof_manager.generate_proof(ONNX_PATH, input_path):
            if proof_manager.verify_proof("proof.json", input_path, "output.json", ONNX_PATH):
                print("✅ Successfully generated and verified proof!")
            else:
                print("❌ Proof verification failed.")
        else:
            print("❌ Proof generation failed.")
            
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()