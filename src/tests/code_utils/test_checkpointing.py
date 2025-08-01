import os
import shutil
import random
import unittest
from pathlib import Path

from tqdm import tqdm
from mypt.code_utils.checkpointing import Checkpointer


class TestCheckpointer(unittest.TestCase):
    def setUp(self):
        """Set up a temporary directory for testing."""
        self.test_dir = Path("./temp_test_checkpoints")
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_save_fn_validation(self):
        """Test that the save_fn signature validation works correctly."""
        
        # --- Invalid Callables ---
        invalid_callables = {
            "missing_both": lambda: None,
            "missing_model": lambda save_dir: None,
            "missing_save_dir": lambda model: None,
            "wrong_kwarg_name": lambda model, directory: None,
        }

        for name, func in invalid_callables.items():
            with self.assertRaises(TypeError, msg=f"Failed for invalid callable: {name}"):
                Checkpointer(root_dir=self.test_dir, save_fn=func)

        # --- Valid Callables ---
        def valid_fn_simple(model, save_dir): pass
        def valid_fn_with_kwargs(model, save_dir, **kwargs): pass
        def valid_fn_with_args(model, save_dir, *args, **kwargs): pass
        
        valid_callables = {
            "simple": valid_fn_simple,
            "with_kwargs": valid_fn_with_kwargs,
            "with_args_and_kwargs": valid_fn_with_args,
        }
        
        for name, func in valid_callables.items():
            try:
                Checkpointer(root_dir=self.test_dir, save_fn=func)
            except TypeError:
                self.fail(f"Valid callable '{name}' incorrectly raised a TypeError.")
    
    def _test_checkpointing_logic(self, mode, identifier_key, rounding_factor):
        """Helper function to test the core checkpointing logic."""
        
        # 1. Outer loop for multiple runs
        for i1 in tqdm(range(100), desc="Testing checkpointing logic"): 
            top_k = random.randint(1, 10)
            num_scores = random.randint(top_k + 5, 50)
            
            # 2. Generate scores and find ground truth
            scores = [random.uniform(0.0, 1000.0) for _ in range(num_scores)]
            sorted_scores = sorted(scores)[:top_k] if mode == 'min' else sorted(scores, reverse=True)[:top_k]
            ground_truth = sorted([round(s, rounding_factor) for s in sorted_scores])
            
            # 3. Inner loop for repeated tests with same data
            for i2 in range(5):
                # Setup
                run_dir = os.path.join(self.test_dir, f"run_{mode}_{identifier_key}_{random.randint(1, 10000)}")
                os.makedirs(run_dir, exist_ok=True)
                
                # Mock save function
                def mock_save(model, save_dir, **kwargs):
                    pass 

                checkpointer = Checkpointer(
                    root_dir=run_dir,
                    save_fn=mock_save,
                    mode=mode,
                    top_k=top_k,
                    identifier_key=identifier_key,
                    rounding_factor=rounding_factor
                )
                
                random.shuffle(scores)
                
                # Save checkpoints
                for i, score in enumerate(scores):
                    checkpointer.save(model=None, score=score, identifier_value=i)
                
                # 4. Assert correctness
                saved_dirs = os.listdir(run_dir)
                
                # Remove the state file from the list
                if "checkpointer_state.json" in saved_dirs:
                    saved_dirs.remove("checkpointer_state.json")

                self.assertEqual(len(saved_dirs), top_k)
                
                saved_scores = []
                for dirname in saved_dirs:
                    # Extract score from directory name like 'checkpoint_epoch_5_score_123.456000'
                    score_str = dirname.split('_score_')[-1]
                    saved_scores.append(float(score_str))
                
                self.assertCountEqual(sorted(saved_scores), ground_truth)

                shutil.rmtree(run_dir)

    def test_checkpointing_with_min_mode(self):
        """Test the checkpointing logic in 'min' mode with different identifiers."""
        self._test_checkpointing_logic(mode='min', identifier_key='epoch', rounding_factor=2)
        shutil.rmtree(self.test_dir)
        self.setUp()
        self._test_checkpointing_logic(mode='min', identifier_key='step', rounding_factor=4)

    def test_checkpointing_with_max_mode(self):
        """Test the checkpointing logic in 'max' mode with different identifiers."""
        self._test_checkpointing_logic(mode='max', identifier_key='fid_score', rounding_factor=5)

    def test_state_saving_and_loading(self):
        """Test that the checkpointer state can be saved and reloaded."""
        top_k = 5
        scores = [10, 20, 5, 30, 2] # top 5 should be [2, 5, 10, 20, 30]
        
        def mock_save(model, save_dir, **kwargs): pass

        # --- First Checkpointer instance ---
        checkpointer1 = Checkpointer(self.test_dir, mock_save, mode='min', top_k=top_k)
        for i, score in enumerate(scores):
            checkpointer1.save(model=None, score=score, identifier_value=i)

        best_score1 = checkpointer1.best_score
        
        # --- Second Checkpointer instance (should load state) ---        
        checkpointer2 = Checkpointer(self.test_dir, mock_save, mode='min', top_k=top_k)
        self.assertEqual(checkpointer1.heap, checkpointer2.heap)
        
        # Add a new best score
        checkpointer2.save(model=None, score=1.0, identifier_value='new_best')
        self.assertNotEqual(best_score1, checkpointer2.best_score)
        self.assertEqual(checkpointer2.best_score, 1.0)
        
        # Add a score that shouldn't be saved
        checkpointer2.save(model=None, score=100.0, identifier_value='worse')
        self.assertEqual(checkpointer2.best_score, 1.0)
        self.assertEqual(len(os.listdir(self.test_dir)) -1, top_k) # -1 for state file

if __name__ == '__main__':
    from mypt.code_utils import pytorch_utils as pu
    pu.seed_everything(42)
    unittest.main()
