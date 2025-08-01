import os
import json
import shutil
import heapq
import inspect
from typing import Any, Callable, List, Tuple

from mypt.shortcuts import P
from mypt.code_utils import directories_and_files as dirf


class Checkpointer:
    """
    Manages saving and retaining the top_k model checkpoints based on a monitored score.

    This class uses a heap to efficiently track the top_k checkpoints, ensuring that
    only the best-performing models are kept on disk. Checkpoints are saved as directories.
    """
    def __init__(self,
                 root_dir: P,
                 save_fn: Callable[..., None],
                 mode: str = 'min',
                 top_k: int = 3,
                 identifier_key: str = 'epoch',
                 rounding_factor: int = 6):
        """
        Initializes the Checkpointer.

        Args:
            root_dir (P): The directory where checkpoint directories will be created.
            save_fn (Callable[..., None]): A function that saves the model. It must accept `model`, `save_dir`,
                                           and any kwargs passed to the `save` method.
            mode (str, optional): One of {'min', 'max'}. In 'min' mode, a lower score is better. Defaults to 'min'.
            top_k (int, optional): The number of best checkpoints to keep. Defaults to 3.
            identifier_key (str, optional): The name of the identifier used in checkpoint directory names. Defaults to 'epoch'.
            rounding_factor (int, optional): The number of decimal places to round the score to in the directory name. Defaults to 6.
        """
        if mode.lower() not in ['min', 'max']:
            raise ValueError("mode must be 'min' or 'max'")

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        if not isinstance(rounding_factor, int) or rounding_factor < 0:
            raise ValueError("rounding_factor must be a non-negative integer")

        self.root_dir = dirf.process_path(root_dir)
        self.save_fn = save_fn
        self.mode = mode.lower()
        self.top_k = top_k
        self.identifier_key = identifier_key
        self.rounding_factor = rounding_factor
        self.heap: List[Tuple[float, str]] = []
        self.state_file = os.path.join(self.root_dir, "checkpointer_state.json")

        self._validate_save_fn()
        self._load_state()

    def _validate_save_fn(self):
        """
        Validates that the save_fn has a signature compatible with the Checkpointer's calling convention.
        """
        try:
            sig = inspect.signature(self.save_fn)
            # Check if the function can be called with 'model' and 'save_dir' as keyword arguments.
            # We don't need to check for **kwargs here, as bind will succeed if they are present.
            sig.bind(model=None, save_dir=None)
        except TypeError as e:
            raise TypeError(
                "The `save_fn` callable must have a signature compatible with `save_fn(model, save_dir, **kwargs)`.\n"
                "It must accept 'model' and 'save_dir' as keyword arguments.\n"
                f"Validation failed with: {e}"
            )

    def save(self, model: Any, score: float, identifier_value: Any, **kwargs) -> None:
        """
        Evaluates a model's score and saves it as a directory if it's among the top_k.

        Args:
            model (Any): The model object to be saved.
            score (float): The score of the model (e.g., validation loss, FID score).
            identifier_value (Any): The value of the identifier for this checkpoint (e.g., 5 for epoch 5).
            **kwargs: Additional keyword arguments to be passed to the `save_fn`.
        """
        heap_score = -score if self.mode == 'min' else score

        if not (len(self.heap) < self.top_k or heap_score > self.heap[0][0]):
            # it means that the current checkpoint is not among the top_k
            return 

        score_for_path = round(score, self.rounding_factor)
        checkpoint_dir = os.path.join(self.root_dir, f"checkpoint_{self.identifier_key}_{identifier_value}_score_{score_for_path}")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.save_fn(model=model, save_dir=checkpoint_dir, **kwargs)
        
        path_to_remove = None
        if len(self.heap) < self.top_k:
            heapq.heappush(self.heap, (heap_score, checkpoint_dir))
        else:
            _, path_to_remove = heapq.heapreplace(self.heap, (heap_score, checkpoint_dir))
        
        if path_to_remove and os.path.exists(path_to_remove):
            shutil.rmtree(path_to_remove)
        
        self._save_state()

    @property
    def best_score(self) -> float:
        """Returns the best score seen so far."""
        if not self.heap:
            return float('-inf') if self.mode == 'max' else float('inf')
        
        best_heap_score = max(self.heap, key=lambda x: x[0])[0]
        return -best_heap_score if self.mode == 'min' else best_heap_score

    @property
    def best_checkpoint_dir(self) -> str:
        """Returns the path to the best checkpoint directory."""
        if not self.heap:
            return None
            
        return max(self.heap, key=lambda x: x[0])[1]

    def _save_state(self):
        """Saves the current state of the heap to a file."""
        with open(self.state_file, 'w') as f:
            json.dump(self.heap, f)

    def _load_state(self):
        """Loads the heap state from a file if it exists."""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                json_heap = json.load(f)
                # convert the json heap to a list of tuples instead of a list of lists
                self.heap = [tuple(l) for l in json_heap]
                heapq.heapify(self.heap)
