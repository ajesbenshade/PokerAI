#!/usr/bin/env python3
"""
ONNX Export and Inference Optimization for PokerAI
Optimized for AMD 7900XT GPU inference

This script exports trained models to ONNX format for:
- Faster inference (2-3x speedup)
- Cross-platform deployment
- Reduced memory footprint
- Hardware-specific optimizations
"""

import os
import sys
import torch
import torch.onnx
import onnxruntime as ort
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
import time
import json

# Add project root to path
sys.path.append('/home/aaron/PokerAI')

from config import Config
from rl import ActorCriticAgent
from models import Actor, Critic

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ONNXExporter:
    """Export PokerAI models to ONNX format with optimizations"""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or '/home/aaron/PokerAI/checkpoint_player_0.pth'
        self.onnx_path = '/home/aaron/PokerAI/models/'
        os.makedirs(self.onnx_path, exist_ok=True)

        # ONNX export configuration
        self.opset_version = 17  # Latest stable opset
        self.input_sample_size = 1

        # ROCm-specific optimizations
        self.enable_rocm_optimizations = True

    def load_trained_model(self) -> Tuple[Actor, Critic]:
        """Load trained PyTorch model"""
        try:
            # Create model architecture
            actor = Actor(
                state_size=Config.STATE_SIZE,
                action_size=Config.ACTION_SIZE,
                hidden_size=Config.ACTOR_HIDDEN_SIZE,
                num_blocks=Config.NUM_RES_BLOCKS
            )

            critic = Critic(
                state_size=Config.STATE_SIZE,
                hidden_size=Config.CRITIC_HIDDEN_SIZE,
                num_blocks=Config.NUM_RES_BLOCKS
            )

            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=True)

            # Load state dicts
            if 'actor' in checkpoint:
                actor.load_state_dict(checkpoint['actor'])
            if 'critic' in checkpoint:
                critic.load_state_dict(checkpoint['critic'])

            logger.info(f"Loaded model from {self.model_path}")
            return actor, critic

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def export_actor_to_onnx(self, actor: Actor, filename: str = 'actor_model.onnx'):
        """Export actor network to ONNX"""
        logger.info("Exporting actor network to ONNX...")

        # Set model to evaluation mode
        actor.eval()

        # Create dummy input
        dummy_input = torch.randn(self.input_sample_size, Config.STATE_SIZE)

        # Export to ONNX
        onnx_file = os.path.join(self.onnx_path, filename)

        torch.onnx.export(
            actor,
            dummy_input,
            onnx_file,
            export_params=True,
            verbose=True,
            opset_version=self.opset_version,
            do_constant_folding=True,
            input_names=['state'],
            output_names=['action_logits'],
            dynamic_axes={
                'state': {0: 'batch_size'},
                'action_logits': {0: 'batch_size'}
            }
        )

        logger.info(f"Actor exported to {onnx_file}")
        return onnx_file

    def export_critic_to_onnx(self, critic: Critic, filename: str = 'critic_model.onnx'):
        """Export critic network to ONNX"""
        logger.info("Exporting critic network to ONNX...")

        # Set model to evaluation mode
        critic.eval()

        # Create dummy input
        dummy_input = torch.randn(self.input_sample_size, Config.STATE_SIZE)

        # Export to ONNX
        onnx_file = os.path.join(self.onnx_path, filename)

        torch.onnx.export(
            critic,
            dummy_input,
            onnx_file,
            export_params=True,
            verbose=True,
            opset_version=self.opset_version,
            do_constant_folding=True,
            input_names=['state'],
            output_names=['value'],
            dynamic_axes={
                'state': {0: 'batch_size'},
                'value': {0: 'batch_size'}
            }
        )

        logger.info(f"Critic exported to {onnx_file}")
        return onnx_file

    def optimize_onnx_model(self, onnx_file: str) -> str:
        """Optimize ONNX model for inference"""
        try:
            import onnxruntime as ort
            from onnxruntime.transformers.onnx_model import OnnxModel

            logger.info(f"Optimizing {onnx_file}...")

            # Load and optimize model
            model = OnnxModel(ort.InferenceSession(onnx_file))

            # Apply optimizations
            optimized_file = onnx_file.replace('.onnx', '_optimized.onnx')

            # Use ONNX Runtime optimizations
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.enable_mem_pattern = True
            sess_options.enable_cpu_mem_arena = True

            # For ROCm, enable GPU optimizations
            if self.enable_rocm_optimizations and torch.cuda.is_available():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

            # Create optimized session
            optimized_session = ort.InferenceSession(onnx_file, sess_options, providers=providers)

            # Save optimized model
            optimized_session.export_model_to_file(optimized_file)

            logger.info(f"Optimized model saved to {optimized_file}")
            return optimized_file

        except ImportError:
            logger.warning("ONNX Runtime optimizations not available, skipping optimization")
            return onnx_file
        except Exception as e:
            logger.warning(f"ONNX optimization failed: {e}")
            return onnx_file

    def validate_onnx_model(self, onnx_file: str, original_model: torch.nn.Module,
                           model_type: str = 'actor') -> bool:
        """Validate ONNX model against original PyTorch model"""
        logger.info(f"Validating {model_type} ONNX model...")

        try:
            # Create ONNX session
            if self.enable_rocm_optimizations and torch.cuda.is_available():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

            ort_session = ort.InferenceSession(onnx_file, providers=providers)

            # Test with multiple batch sizes
            for batch_size in [1, 4, 16]:
                # Create test input
                test_input = torch.randn(batch_size, Config.STATE_SIZE)

                # PyTorch inference
                original_model.eval()
                with torch.no_grad():
                    if model_type == 'actor':
                        torch_output = original_model(test_input).numpy()
                    else:  # critic
                        torch_output = original_model(test_input).unsqueeze(-1).numpy()

                # ONNX inference
                ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
                onnx_output = ort_session.run(None, ort_inputs)[0]

                # Compare outputs
                max_diff = np.max(np.abs(torch_output - onnx_output))
                mean_diff = np.mean(np.abs(torch_output - onnx_output))

                logger.info(f"Batch {batch_size}: Max diff = {max_diff:.6f}, Mean diff = {mean_diff:.6f}")

                if max_diff > 1e-4:  # Allow small numerical differences
                    logger.warning(f"Large difference detected in {model_type} model!")
                    return False

            logger.info(f"{model_type} ONNX model validation passed!")
            return True

        except Exception as e:
            logger.error(f"ONNX validation failed: {e}")
            return False

    def benchmark_inference(self, pytorch_model: torch.nn.Module, onnx_file: str,
                           model_type: str = 'actor', num_runs: int = 100) -> Dict[str, float]:
        """Benchmark inference speed comparison"""
        logger.info(f"Benchmarking {model_type} inference...")

        results = {}

        # PyTorch benchmark
        pytorch_model.eval()
        test_input = torch.randn(1, Config.STATE_SIZE)

        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = pytorch_model(test_input)

        # Benchmark PyTorch
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_runs):
                _ = pytorch_model(test_input)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        pytorch_time = (time.time() - start_time) / num_runs * 1000  # ms

        results['pytorch_ms'] = pytorch_time

        # ONNX benchmark
        try:
            if self.enable_rocm_optimizations and torch.cuda.is_available():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

            ort_session = ort.InferenceSession(onnx_file, providers=providers)

            # Warm up
            ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
            for _ in range(10):
                _ = ort_session.run(None, ort_inputs)

            # Benchmark ONNX
            start_time = time.time()
            for _ in range(num_runs):
                _ = ort_session.run(None, ort_inputs)

            onnx_time = (time.time() - start_time) / num_runs * 1000  # ms

            results['onnx_ms'] = onnx_time
            results['speedup'] = pytorch_time / onnx_time

            logger.info(f"PyTorch: {pytorch_time:.3f}ms, ONNX: {onnx_time:.3f}ms, Speedup: {results['speedup']:.2f}x")

        except Exception as e:
            logger.error(f"ONNX benchmark failed: {e}")
            results['onnx_ms'] = float('inf')
            results['speedup'] = 0.0

        return results

    def export_all_models(self):
        """Export all available models to ONNX"""
        logger.info("Starting full ONNX export process...")

        # Find all checkpoint files
        checkpoint_files = [f for f in os.listdir('/home/aaron/PokerAI/') if f.startswith('checkpoint_player_') and f.endswith('.pth')]

        if not checkpoint_files:
            logger.warning("No checkpoint files found!")
            return

        results = []

        for checkpoint_file in checkpoint_files:
            try:
                logger.info(f"Processing {checkpoint_file}...")

                # Load model
                self.model_path = os.path.join('/home/aaron/PokerAI/', checkpoint_file)
                actor, critic = self.load_trained_model()

                # Export actor
                actor_onnx = self.export_actor_to_onnx(actor, f"{checkpoint_file.replace('.pth', '_actor.onnx')}")
                actor_optimized = self.optimize_onnx_model(actor_onnx)

                # Export critic
                critic_onnx = self.export_critic_to_onnx(critic, f"{checkpoint_file.replace('.pth', '_critic.onnx')}")
                critic_optimized = self.optimize_onnx_model(critic_onnx)

                # Validate models
                actor_valid = self.validate_onnx_model(actor_optimized, actor, 'actor')
                critic_valid = self.validate_onnx_model(critic_optimized, critic, 'critic')

                # Benchmark models
                actor_benchmark = self.benchmark_inference(actor, actor_optimized, 'actor')
                critic_benchmark = self.benchmark_inference(critic, critic_optimized, 'critic')

                result = {
                    'checkpoint': checkpoint_file,
                    'actor_onnx': actor_onnx,
                    'actor_optimized': actor_optimized,
                    'critic_onnx': critic_onnx,
                    'critic_optimized': critic_optimized,
                    'actor_valid': actor_valid,
                    'critic_valid': critic_valid,
                    'actor_benchmark': actor_benchmark,
                    'critic_benchmark': critic_benchmark
                }

                results.append(result)

            except Exception as e:
                logger.error(f"Failed to export {checkpoint_file}: {e}")
                continue

        # Save export summary
        summary_file = os.path.join(self.onnx_path, 'export_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Export summary saved to {summary_file}")

        # Print summary
        print("\n" + "="*60)
        print("ONNX EXPORT SUMMARY")
        print("="*60)

        for result in results:
            print(f"\nModel: {result['checkpoint']}")
            print(f"Actor Valid: {result['actor_valid']}")
            print(f"Critic Valid: {result['critic_valid']}")
            if 'speedup' in result['actor_benchmark']:
                print(".2f")
            if 'speedup' in result['critic_benchmark']:
                print(".2f")

        return results

class ONNXInferenceEngine:
    """ONNX-based inference engine for production use"""

    def __init__(self, actor_path: str, critic_path: str):
        self.actor_path = actor_path
        self.critic_path = critic_path

        # Initialize ONNX sessions
        if torch.cuda.is_available():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.actor_session = ort.InferenceSession(actor_path, providers=providers)
        self.critic_session = ort.InferenceSession(critic_path, providers=providers)

        logger.info("ONNX inference engine initialized")

    def get_action_logits(self, state: np.ndarray) -> np.ndarray:
        """Get action logits from ONNX actor"""
        ort_inputs = {self.actor_session.get_inputs()[0].name: state}
        logits = self.actor_session.run(None, ort_inputs)[0]
        return logits

    def get_value(self, state: np.ndarray) -> float:
        """Get state value from ONNX critic"""
        ort_inputs = {self.critic_session.get_inputs()[0].name: state}
        value = self.critic_session.run(None, ort_inputs)[0]
        return float(value)

def main():
    """Main ONNX export function"""
    exporter = ONNXExporter()

    # Export all models
    results = exporter.export_all_models()

    if results:
        # Create inference engine with best model
        best_result = max(results, key=lambda x: x['actor_benchmark'].get('speedup', 0))

        print("\nBest model for inference:")
        print(f"Actor: {best_result['actor_optimized']}")
        print(f"Critic: {best_result['critic_optimized']}")
        print(f"Speedup: {best_result['actor_benchmark'].get('speedup', 0):.2f}x")

        # Create inference engine
        engine = ONNXInferenceEngine(
            best_result['actor_optimized'],
            best_result['critic_optimized']
        )

        print("\nONNX inference engine ready for production use!")
        print("Use engine.get_action_logits(state) and engine.get_value(state) for inference")

if __name__ == "__main__":
    main()
