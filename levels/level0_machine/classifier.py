"""Level 0 Neural Classifier: Byte → Mnemonic.

Trained transformer model that predicts instruction mnemonics from raw bytes.
Achieves 100% accuracy on the gate test.

Usage:
    from levels.level0_machine.classifier import Level0Classifier
    
    classifier = Level0Classifier()
    mnemonic = classifier.predict(b'\\x55')  # Returns 'push'
    mnemonic = classifier.predict('4883ec20')  # Returns 'sub'
"""

from pathlib import Path

import torch
import torch.nn as nn

# Model configuration
MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "level0"
MAX_LEN = 15


class ByteClassifier(nn.Module):
    """Transformer-based byte classifier."""

    def __init__(
        self,
        num_classes: int,
        max_len: int = MAX_LEN,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
    ):
        super().__init__()
        self.byte_embed = nn.Embedding(256, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_classes),
        )
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        embeddings = self.byte_embed(x) + self.pos_embed(positions)
        encoded = self.transformer(embeddings)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)


class Level0Classifier:
    """Neural classifier for x86-64 instruction bytes → mnemonic.

    This classifier achieves 100% accuracy on the real-world gate test.
    It uses a small transformer model (~860K params) trained on Capstone
    ground truth from real system binaries.

    Attributes:
        num_classes: Number of mnemonic classes (136)
        mnemonics: List of known mnemonics
    """

    def __init__(self, model_dir: Path | str | None = None):
        """Load the trained model.

        Args:
            model_dir: Path to model directory. Defaults to models/level0/
        """
        model_dir = Path(model_dir) if model_dir else MODEL_DIR

        checkpoint_path = model_dir / "model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model not found at {checkpoint_path}. "
                "Run notebooks/train_level0.py to train the model."
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.mnemonic_to_id: dict[str, int] = checkpoint["mnemonic_to_id"]
        self.id_to_mnemonic: dict[int, str] = checkpoint["id_to_mnemonic"]
        self.num_classes: int = checkpoint["num_classes"]
        self.max_len: int = checkpoint["max_len"]

        self.model = ByteClassifier(self.num_classes, self.max_len)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    @property
    def mnemonics(self) -> list[str]:
        """List of known mnemonics."""
        return sorted(self.mnemonic_to_id.keys())

    def predict(self, bytes_input: bytes | str | list[int]) -> str:
        """Predict mnemonic from instruction bytes.

        Args:
            bytes_input: Raw bytes, hex string, or list of byte values

        Returns:
            Predicted mnemonic (e.g., 'push', 'mov', 'call')

        Examples:
            >>> classifier.predict(b'\\x55')
            'push'
            >>> classifier.predict('4883ec20')
            'sub'
            >>> classifier.predict([0xc3])
            'ret'
        """
        # Convert input to list of integers
        if isinstance(bytes_input, str):
            bytes_input = bytes_input.replace(" ", "")
            bytes_list = [int(bytes_input[i : i + 2], 16) for i in range(0, len(bytes_input), 2)]
        elif isinstance(bytes_input, bytes):
            bytes_list = list(bytes_input)
        else:
            bytes_list = list(bytes_input)

        # Pad or truncate to max_len
        if len(bytes_list) < self.max_len:
            bytes_list = bytes_list + [0] * (self.max_len - len(bytes_list))
        else:
            bytes_list = bytes_list[: self.max_len]

        # Predict
        x = torch.tensor([bytes_list], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(x)
            pred_id = logits.argmax(dim=1).item()

        return self.id_to_mnemonic[pred_id]

    def predict_batch(self, byte_sequences: list[bytes | str | list[int]]) -> list[str]:
        """Predict mnemonics for multiple instruction bytes.

        Args:
            byte_sequences: List of byte inputs

        Returns:
            List of predicted mnemonics
        """
        return [self.predict(seq) for seq in byte_sequences]

    def is_known(self, mnemonic: str) -> bool:
        """Check if a mnemonic is in the model's vocabulary."""
        return mnemonic.lower() in self.mnemonic_to_id


# Convenience function
def predict_mnemonic(bytes_input: bytes | str | list[int]) -> str:
    """Predict mnemonic from instruction bytes.

    This is a convenience function that creates a classifier on first call.
    For repeated predictions, create a Level0Classifier instance directly.

    Args:
        bytes_input: Raw bytes, hex string, or list of byte values

    Returns:
        Predicted mnemonic
    """
    global _classifier
    if "_classifier" not in globals():
        _classifier = Level0Classifier()
    return _classifier.predict(bytes_input)
