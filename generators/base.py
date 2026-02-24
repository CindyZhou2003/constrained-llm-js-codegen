from abc import ABC, abstractmethod
from typing import List, Optional

class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str, stop_tokens: Optional[List[str]] = None, **kwargs) -> str:
        pass

    def _post_process_stop(self, text: str, stop_tokens: List[str] | None) -> str:
        if not stop_tokens: return text
        min_stop_index = len(text)
        found = False
        for stop in stop_tokens:
            idx = text.find(stop)
            if idx != -1:
                min_stop_index = min(min_stop_index, idx)
                found = True
        return text[:min_stop_index] if found else text