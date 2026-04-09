from abc import ABC, abstractmethod
from typing import List, Optional

class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str, stop_tokens: Optional[List[str]] = None, **kwargs) -> str:
        pass

    # Stop tokens that are only valid at the top level (brace depth == 0).
    # Inside a function body these would falsely truncate the completion.
    _DEPTH_SENSITIVE_STOPS = {"\n//", "\n/*"}

    def _post_process_stop(self, text: str, stop_tokens: List[str] | None, initial_depth: int = 0) -> str:
        if not stop_tokens:
            return text

        # Pre-compute brace depth at every character position so we can
        # skip depth-sensitive stop tokens while still inside a function body.
        depth = initial_depth
        depth_at = []
        for ch in text:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth = max(depth - 1, 0)
            depth_at.append(depth)

        min_stop_index = len(text)
        found = False
        for stop in stop_tokens:
            start = 0
            while True:
                idx = text.find(stop, start)
                if idx == -1:
                    break
                # Depth-sensitive tokens are skipped when inside a block.
                if stop in self._DEPTH_SENSITIVE_STOPS and depth_at[idx] > 0:
                    start = idx + 1
                    continue
                min_stop_index = min(min_stop_index, idx)
                found = True
                break
        return text[:min_stop_index] if found else text