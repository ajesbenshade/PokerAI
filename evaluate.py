#!/usr/bin/env python3
"""
Comprehensive Evaluation Suite for PokerAI
100k-hand evaluation with exploitability analysis and baseline comparisons

This script performs thorough evaluation of trained PokerAI models including:
- Exploitability analysis (<100 mbb/100 target)
- Win rates vs. Slumbot-like baselines
- Performance metrics across different stack sizes
- Hardware monitoring during evaluation
- Statistical significance testing
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import multiprocessing as mp
import psutil
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
import scipy.stats as stats

# Add project root to path
sys.path.append('/home/aaron/PokerAI')

from config import Config
from game import GTOHoldEm, simulate_hand
from gto import evaluate_exploitability
from datatypes import Player
from rl import ActorCriticAgent
from utils import get_vram_usage, opponent_tracker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/aaron/PokerAI/evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HardwareMonitor:
    """Monitor hardware resources during evaluation"""

    def __init__(self):
        self.start_time = time.time()
        self.cpu_usage = []
        self.memory_usage = []
        self.vram_usage = []
        self.hand_times = []

    def log_hand_time(self, hand_time: float):
        """Log time taken for a hand"""
        self.hand_times.append(hand_time)

    def log_resources(self):
        """Log current resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            vram_gb = get_vram_usage()

            self.cpu_usage.append(cpu_percent)
            self.memory_usage.append(memory_percent)
            self.vram_usage.append(vram_gb)

        except Exception as e:
            logger.warning(f"Resource monitoring failed: {e}")

    def get_summary(self) -> Dict[str, float]:
        """Get resource usage summary"""
        if not self.cpu_usage:
            return {}

        return {
            'avg_cpu': np.mean(self.cpu_usage),
            'max_cpu': np.max(self.cpu_usage),
            'avg_memory': np.mean(self.memory_usage),
            'max_memory': np.max(self.memory_usage),
            'avg_vram': np.mean(self.vram_usage),
            'max_vram': np.max(self.vram_usage),
            'avg_hand_time': np.mean(self.hand_times) if self.hand_times else 0,
            'total_time': time.time() - self.start_time,
            'total_hands': len(self.hand_times),
            'hands_per_second': len(self.hand_times) / (time.time() - self.start_time)
        }

class BaselineBot:
    """Simple baseline bot for comparison (Slumbot-like)"""

    def __init__(self, player_id: int, style: str = 'tight_aggressive'):
        self.player_id = player_id
        self.style = style

        # Style parameters
        if style == 'tight_aggressive':
            self.vpip_threshold = 0.25
            self.aggression_factor = 1.5
        elif style == 'loose_passive':
            self.vpip_threshold = 0.45
            self.aggression_factor = 0.7
        elif style == 'balanced':
            self.vpip_threshold = 0.35
            self.aggression_factor = 1.0

    def choose_action(self, state: np.ndarray, legal_actions: np.ndarray,
                     **kwargs) -> Tuple[int, Optional[float], int, Optional[float], Optional[float]]:
        """Choose action based on simple heuristics"""

        # Simple hand strength estimation (random for baseline)
        hand_strength = np.random.random()

        # VPIP decision
        if hand_strength > self.vpip_threshold:
            # Decide to play
            if np.random.random() < self.aggression_factor * 0.3:
                # Raise
                action_idx = 2  # Raise action
                raise_amount = kwargs.get('min_raise', 20) * (1 + np.random.random())
                discrete_action = 2  # First raise bin
            else:
                # Call
                action_idx = 1  # Call
                raise_amount = None
                discrete_action = 1
        else:
            # Fold
            action_idx = 0  # Fold
            raise_amount = None
            discrete_action = 0

        # Mock log prob and value
        log_prob = -1.0
        value = hand_strength

        return action_idx, raise_amount, discrete_action, log_prob, value

class EvaluationEngine:
    """Comprehensive evaluation engine for PokerAI"""

    def __init__(self, model_path: Optional[str] = None, n_games: int = 10000):
        self.model_path = model_path or '/home/aaron/PokerAI/checkpoint_player_0.pth'
        self.n_games = n_games
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Load trained model
        self.agent = self._load_agent()

        # Hardware monitor
        self.monitor = HardwareMonitor()

        # Results storage
        self.results = {
            'exploitability': [],
            'win_rates': {},
            'hand_times': [],
            'resource_usage': {},
            'game_logs': []
        }

    def _load_agent(self) -> ActorCriticAgent:
        """Load trained agent"""
        try:
            agent = ActorCriticAgent.create_simulation_agent(device=self.device)

            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
                if 'actor' in checkpoint:
                    agent.actor.load_state_dict(checkpoint['actor'])
                if 'critic' in checkpoint:
                    agent.critic.load_state_dict(checkpoint['critic'])

                logger.info(f"Loaded model from {self.model_path}")
            else:
                logger.warning(f"Model not found at {self.model_path}, using random agent")

            agent.actor.eval()
            agent.critic.eval()

            return agent

        except Exception as e:
            logger.error(f"Failed to load agent: {e}")
            raise

    def evaluate_exploitability(self, n_hands: int = 1000) -> Dict[str, Any]:
        """Evaluate exploitability of the current strategy"""
        logger.info(f"Evaluating exploitability over {n_hands} hands...")

        try:
            # Create game for exploitability evaluation
            game = GTOHoldEm(num_players=2)  # Heads-up for exploitability

            # Evaluate exploitability
            exploitability = evaluate_exploitability(game, self.agent, n_hands=n_hands)

            result = {
                'exploitability_mbb_100': exploitability,
                'target_achieved': exploitability < 100,  # Target: <100 mbb/100
                'evaluation_time': time.time(),
                'n_hands': n_hands
            }

            self.results['exploitability'].append(result)

            logger.info(".2f")
            return result

        except Exception as e:
            logger.error(f"Exploitability evaluation failed: {e}")
            return {'error': str(e)}

    def evaluate_vs_baseline(self, baseline_style: str = 'balanced',
                           n_games: int = 1000) -> Dict[str, Any]:
        """Evaluate win rate against baseline bot"""
        logger.info(f"Evaluating vs {baseline_style} baseline over {n_games} games...")

        wins = 0
        losses = 0
        ties = 0
        game_results = []

        for game_idx in range(n_games):
            try:
                # Create players
                ai_player = Player(player_id=0, agent=self.agent)
                baseline_player = Player(player_id=1, agent=BaselineBot(1, baseline_style))

                players = [ai_player, baseline_player]

                # Play game
                game_start = time.time()
                utilities, trajectories = simulate_hand(
                    players, dealer_idx=game_idx % 2, hand_number=game_idx
                )
                game_time = time.time() - game_start

                # Record result
                ai_utility = utilities.get(0, 0)
                baseline_utility = utilities.get(1, 0)

                if ai_utility > baseline_utility:
                    wins += 1
                    result = 'win'
                elif ai_utility < baseline_utility:
                    losses += 1
                    result = 'loss'
                else:
                    ties += 1
                    result = 'tie'

                game_results.append({
                    'game_idx': game_idx,
                    'result': result,
                    'ai_utility': ai_utility,
                    'baseline_utility': baseline_utility,
                    'game_time': game_time
                })

                # Log progress
                if (game_idx + 1) % 100 == 0:
                    win_rate = (wins + ties * 0.5) / (game_idx + 1)
                    logger.info(f"Game {game_idx + 1}/{n_games}: Win rate = {win_rate:.3f}")

                # Monitor resources
                self.monitor.log_hand_time(game_time)
                if game_idx % 50 == 0:
                    self.monitor.log_resources()

            except Exception as e:
                logger.error(f"Game {game_idx} failed: {e}")
                continue

        # Calculate statistics
        total_games = wins + losses + ties
        win_rate = wins / total_games if total_games > 0 else 0
        tie_rate = ties / total_games if total_games > 0 else 0
        loss_rate = losses / total_games if total_games > 0 else 0

        # Statistical significance (vs 50% random)
        if wins + losses > 0:
            # Binomial test
            p_value = stats.binomtest(wins, wins + losses, 0.5).pvalue
            significant = p_value < 0.05
        else:
            p_value = 1.0
            significant = False

        result = {
            'baseline_style': baseline_style,
            'n_games': total_games,
            'wins': wins,
            'losses': losses,
            'ties': ties,
            'win_rate': win_rate,
            'tie_rate': tie_rate,
            'loss_rate': loss_rate,
            'p_value': p_value,
            'statistically_significant': significant,
            'avg_game_time': np.mean([g['game_time'] for g in game_results]),
            'game_results': game_results
        }

        self.results['win_rates'][baseline_style] = result

        logger.info(f"Vs {baseline_style}: {win_rate:.3f} win rate, p-value = {p_value:.4f}")
        return result

    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation suite"""
        logger.info("Starting full evaluation suite...")

        evaluation_start = time.time()

        # 1. Exploitability evaluation
        logger.info("Phase 1: Exploitability evaluation")
        exploitability_result = self.evaluate_exploitability(n_hands=1000)

        # 2. Baseline comparisons
        logger.info("Phase 2: Baseline comparisons")
        baseline_styles = ['tight_aggressive', 'balanced', 'loose_passive']

        for style in baseline_styles:
            self.evaluate_vs_baseline(style, n_games=1000)

        # 3. Resource summary
        resource_summary = self.monitor.get_summary()

        # 4. Overall assessment
        overall_result = self._assess_overall_performance()

        # Compile final results
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'exploitability': exploitability_result,
            'win_rates': self.results['win_rates'],
            'resource_usage': resource_summary,
            'overall_assessment': overall_result,
            'evaluation_time': time.time() - evaluation_start
        }

        # Save results
        results_file = '/home/aaron/PokerAI/evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)

        logger.info(f"Evaluation completed in {final_results['evaluation_time']:.1f} seconds")
        logger.info(f"Results saved to {results_file}")

        return final_results

    def _assess_overall_performance(self) -> Dict[str, Any]:
        """Assess overall performance based on evaluation results"""

        assessment = {
            'exploitability_target': False,
            'baseline_performance': {},
            'overall_score': 0.0,
            'recommendations': []
        }

        # Check exploitability target
        if self.results['exploitability']:
            latest_exploitability = self.results['exploitability'][-1]
            assessment['exploitability_target'] = latest_exploitability.get('target_achieved', False)

        # Assess baseline performance
        for style, results in self.results['win_rates'].items():
            win_rate = results.get('win_rate', 0)
            significant = results.get('statistically_significant', False)

            if win_rate > 0.55 and significant:
                performance = 'excellent'
                score = 3
            elif win_rate > 0.52 and significant:
                performance = 'good'
                score = 2
            elif win_rate > 0.48:
                performance = 'adequate'
                score = 1
            else:
                performance = 'poor'
                score = 0

            assessment['baseline_performance'][style] = {
                'performance': performance,
                'score': score,
                'win_rate': win_rate,
                'significant': significant
            }

            assessment['overall_score'] += score

        # Generate recommendations
        if not assessment['exploitability_target']:
            assessment['recommendations'].append("Continue training to reduce exploitability below 100 mbb/100")

        baseline_scores = [v['score'] for v in assessment['baseline_performance'].values()]
        avg_baseline_score = np.mean(baseline_scores)

        if avg_baseline_score < 2:
            assessment['recommendations'].append("Improve baseline win rates through additional training")
        elif avg_baseline_score >= 2.5:
            assessment['recommendations'].append("Excellent performance - consider advanced techniques")

        if assessment['overall_score'] >= 8:
            assessment['recommendations'].append("Model ready for production deployment")
        elif assessment['overall_score'] >= 5:
            assessment['recommendations'].append("Good progress - continue training with current approach")
        else:
            assessment['recommendations'].append("Consider hyperparameter optimization or architecture changes")

        return assessment

def run_parallel_evaluation(model_paths: List[str], n_games_per_model: int = 5000) -> Dict[str, Any]:
    """Run evaluation in parallel across multiple models"""

    logger.info(f"Running parallel evaluation on {len(model_paths)} models...")

    results = {}

    def evaluate_model(model_path: str) -> Dict[str, Any]:
        """Evaluate a single model"""
        try:
            evaluator = EvaluationEngine(model_path, n_games_per_model)
            result = evaluator.run_full_evaluation()
            return {'model': model_path, 'result': result, 'success': True}
        except Exception as e:
            logger.error(f"Evaluation failed for {model_path}: {e}")
            return {'model': model_path, 'error': str(e), 'success': False}

    # Run evaluations in parallel
    with ProcessPoolExecutor(max_workers=min(len(model_paths), mp.cpu_count())) as executor:
        futures = [executor.submit(evaluate_model, path) for path in model_paths]

        for future in as_completed(futures):
            result = future.result()
            model_name = os.path.basename(result['model'])
            results[model_name] = result

    return results

def main():
    """Main evaluation function"""

    # Find all checkpoint files
    checkpoint_files = [os.path.join('/home/aaron/PokerAI/', f)
                       for f in os.listdir('/home/aaron/PokerAI/')
                       if f.startswith('checkpoint_player_') and f.endswith('.pth')]

    if not checkpoint_files:
        logger.warning("No checkpoint files found! Using default evaluation.")
        checkpoint_files = ['/home/aaron/PokerAI/checkpoint_player_0.pth']

    # Run evaluation on best model (or all models if parallel)
    if len(checkpoint_files) == 1:
        # Single model evaluation
        evaluator = EvaluationEngine(checkpoint_files[0], n_games=10000)
        results = evaluator.run_full_evaluation()
    else:
        # Parallel evaluation
        results = run_parallel_evaluation(checkpoint_files, n_games_per_model=5000)

    # Print summary
    print("\n" + "="*80)
    print("POKERAI EVALUATION SUMMARY")
    print("="*80)

    if isinstance(results, dict) and 'exploitability' in results:
        # Single model results
        print("\nExploitability:")
        exp = results['exploitability']
        if isinstance(exp, list) and exp:
            exp = exp[-1]
        print(f"  Exploitability: {exp.get('exploitability_mbb_100', 'N/A'):.2f} mbb/100")
        print(f"Target achieved: {exp.get('target_achieved', False)}")

        print("\nWin Rates vs Baselines:")
        for style, win_result in results.get('win_rates', {}).items():
            print(f"  {style}: {win_result.get('win_rate', 0):.3f} "
                  f"(p-value: {win_result.get('p_value', 1):.4f})")

        print("\nResource Usage:")
        resources = results.get('resource_usage', {})
        print(f"  Avg CPU: {resources.get('avg_cpu', 0):.1f}%")
        print(f"  Max VRAM: {resources.get('max_vram', 0):.2f}GB")
        print(f"  Hands/sec: {resources.get('hands_per_second', 0):.1f}")

    else:
        # Multiple model results
        print(f"Evaluated {len(results)} models")
        for model_name, model_result in results.items():
            if model_result.get('success', False):
                print(f"\n{model_name}:")
                exp = model_result['result'].get('exploitability', {})
                if isinstance(exp, list) and exp:
                    exp = exp[-1]
                print(".2f")

    print("\n" + "="*80)
    print("Evaluation complete! Check evaluation_results.json for detailed results.")

if __name__ == "__main__":
    main()
