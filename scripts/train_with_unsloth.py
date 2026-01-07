"""
Nano-Analyst Fine-Tuning Script with Unsloth
=============================================
Trains Llama-3-8B on Spider dataset using QLoRA for efficient Text-to-SQL generation.

Optimized for:
- Google Colab T4 GPU (15GB VRAM)
- Memory-efficient 4-bit quantization
- 2x faster training with Unsloth

Author: Senior AI Architect
Model: Llama-3-8B-Instruct (Quantized)
Technique: QLoRA (Quantized Low-Rank Adaptation)
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training hyperparameters optimized for T4 GPU."""

    # Model
    base_model: str = "unsloth/llama-3-8b-Instruct-bnb-4bit"
    max_seq_length: int = 1536  # Schema + Question + SQL
    load_in_4bit: bool = True

    # LoRA
    lora_r: int = 32  # Rank (higher = more capacity, but slower)
    lora_alpha: int = 64  # Scaling factor (typically 2x rank)
    lora_dropout: float = 0.05
    target_modules: List[str] = None

    # Training
    per_device_train_batch_size: int = 2  # T4 limit
    gradient_accumulation_steps: int = 8  # Effective batch = 16
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    max_steps: int = -1  # -1 means use num_epochs
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 250

    # Optimization
    optim: str = "adamw_8bit"  # Memory-efficient optimizer
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"

    # Mixed Precision
    fp16: bool = False
    bf16: bool = False  # Set to True if GPU supports it (A100, H100)

    # Paths
    data_dir: str = None
    output_dir: str = None

    def __post_init__(self):
        """Set default target modules for Llama-3."""
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

        # Auto-detect bf16 support
        if torch.cuda.is_available():
            # A100, H100, or newer support bfloat16
            compute_capability = torch.cuda.get_device_capability()
            if compute_capability[0] >= 8:  # Ampere or newer
                self.bf16 = True
                logger.info("✓ GPU supports BF16, enabling for better stability")
            else:
                self.fp16 = True
                logger.info("✓ Using FP16 (GPU doesn't support BF16)")


class NanoAnalystTrainer:
    """Handles fine-tuning of Llama-3 for Text-to-SQL using Unsloth."""

    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup_environment(self):
        """Check GPU and install dependencies."""
        logger.info("=" * 70)
        logger.info("ENVIRONMENT SETUP")
        logger.info("=" * 70)

        # Check CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("❌ No GPU detected! Please enable GPU in Colab: Runtime → Change runtime type → GPU")

        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"✓ GPU: {gpu_name}")
        logger.info(f"✓ VRAM: {gpu_memory:.1f} GB")
        logger.info(f"✓ CUDA Version: {torch.version.cuda}")

        # Check for required libraries
        try:
            import unsloth
            logger.info("✓ Unsloth installed")
        except ImportError:
            raise ImportError(
                "❌ Unsloth not installed. Run:\n"
                "pip install unsloth"
            )

        logger.info("=" * 70 + "\n")

    def load_model_and_tokenizer(self):
        """Load quantized Llama-3 model with Unsloth."""
        logger.info("=" * 70)
        logger.info("LOADING MODEL")
        logger.info("=" * 70)

        from unsloth import FastLanguageModel

        logger.info(f"Loading {self.config.base_model}...")
        logger.info("This may take 2-3 minutes on first run (downloads ~5GB)...")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.base_model,
            max_seq_length=self.config.max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=self.config.load_in_4bit,
        )

        logger.info("✓ Model loaded successfully")

        # Add LoRA adapters
        logger.info("\nAdding LoRA adapters...")
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
            use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
            random_state=42,
        )

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_pct = 100 * trainable_params / total_params

        logger.info(f"✓ LoRA adapters added")
        logger.info(f"  Trainable params: {trainable_params:,} ({trainable_pct:.2f}%)")
        logger.info(f"  Total params: {total_params:,}")
        logger.info("=" * 70 + "\n")

    def format_prompt(self, example: Dict) -> str:
        """
        Format training example into Llama-3 chat template.

        Args:
            example: Dictionary with 'instruction', 'input', 'output'

        Returns:
            Formatted prompt string
        """
        # Llama-3-Instruct uses special tokens
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{example['instruction']}<|eot_id|><|start_header_id|>user<|end_header_id|>

{example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example['output']}<|eot_id|>"""
        return prompt

    def load_and_prepare_data(self):
        """Load Spider dataset and prepare for training."""
        logger.info("=" * 70)
        logger.info("LOADING DATA")
        logger.info("=" * 70)

        data_dir = Path(self.config.data_dir)

        # Load train and validation data
        train_path = data_dir / "train.json"
        val_path = data_dir / "validation.json"

        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found at {train_path}")

        with open(train_path, 'r') as f:
            train_data = json.load(f)

        with open(val_path, 'r') as f:
            val_data = json.load(f)

        logger.info(f"✓ Loaded {len(train_data)} training examples")
        logger.info(f"✓ Loaded {len(val_data)} validation examples")

        # Format prompts
        logger.info("\nFormatting prompts...")
        train_prompts = [self.format_prompt(ex) for ex in train_data]
        val_prompts = [self.format_prompt(ex) for ex in val_data]

        # Create datasets
        from datasets import Dataset

        train_dataset = Dataset.from_dict({"text": train_prompts})
        val_dataset = Dataset.from_dict({"text": val_prompts})

        logger.info("✓ Datasets formatted")
        logger.info("=" * 70 + "\n")

        return train_dataset, val_dataset

    def train(self, train_dataset, val_dataset):
        """Execute training loop."""
        logger.info("=" * 70)
        logger.info("TRAINING")
        logger.info("=" * 70)

        from trl import SFTTrainer
        from transformers import TrainingArguments

        # Prepare output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_train_epochs,
            max_steps=self.config.max_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            optim=self.config.optim,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            lr_scheduler_type=self.config.lr_scheduler_type,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_dir=str(output_dir / "logs"),
            report_to="none",  # Disable WandB for now (can enable later)
            save_total_limit=3,  # Keep only 3 best checkpoints
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            args=training_args,
            packing=False,  # Don't pack sequences (important for SQL)
        )

        # Disable cache for training
        self.model.config.use_cache = False

        logger.info("Starting training...")
        logger.info(f"  Total epochs: {self.config.num_train_epochs}")
        logger.info(f"  Batch size: {self.config.per_device_train_batch_size}")
        logger.info(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps}")
        logger.info(f"  Learning rate: {self.config.learning_rate}")
        logger.info("=" * 70 + "\n")

        # Train
        self.trainer.train()

        logger.info("\n" + "=" * 70)
        logger.info("✓ TRAINING COMPLETE!")
        logger.info("=" * 70)

    def save_model(self):
        """Save the fine-tuned model."""
        logger.info("\nSaving model...")

        output_dir = Path(self.config.output_dir)

        # Save LoRA adapters (lightweight)
        lora_dir = output_dir / "lora_adapters"
        lora_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(str(lora_dir))
        self.tokenizer.save_pretrained(str(lora_dir))

        logger.info(f"✓ LoRA adapters saved to: {lora_dir}")

        # Optionally save merged model (for deployment)
        # This merges LoRA weights into base model
        # Uncomment if you want full model (takes ~16GB disk space)
        """
        merged_dir = output_dir / "merged_model"
        merged_dir.mkdir(parents=True, exist_ok=True)

        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(self.model)  # Enable inference mode
        self.model.save_pretrained_merged(
            str(merged_dir),
            self.tokenizer,
            save_method="merged_16bit"
        )
        logger.info(f"✓ Merged model saved to: {merged_dir}")
        """

    def run_full_pipeline(self):
        """Execute complete training pipeline."""
        try:
            self.setup_environment()
            self.load_model_and_tokenizer()
            train_dataset, val_dataset = self.load_and_prepare_data()
            self.train(train_dataset, val_dataset)
            self.save_model()

            logger.info("\n" + "=" * 70)
            logger.info("🎉 SUCCESS! Nano-Analyst training complete!")
            logger.info("=" * 70)
            logger.info(f"Model saved to: {self.config.output_dir}")
            logger.info("\nNext steps:")
            logger.info("1. Test the model on validation set")
            logger.info("2. Build the SQLAgent with RAG (Step 3)")
            logger.info("3. Run evaluation benchmarks (Step 4)")
            logger.info("=" * 70)

        except Exception as e:
            logger.error(f"\n❌ Training failed with error: {e}")
            raise


def main():
    """Main execution function."""

    # Configure paths (adjust for your environment)
    if os.path.exists('/content'):  # Google Colab
        base_dir = Path('/content/drive/MyDrive/nano-analyst')
    else:  # Local
        base_dir = Path.home() / 'nano-analyst'

    config = TrainingConfig(
        data_dir=str(base_dir / 'data' / 'processed'),
        output_dir=str(base_dir / 'models' / 'nano-analyst-v1'),
    )

    trainer = NanoAnalystTrainer(config)
    trainer.run_full_pipeline()


if __name__ == "__main__":
    main()
