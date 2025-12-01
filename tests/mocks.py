import os
from typing import Any

from llamda.utils.llm_client.base import BaseClient, BaseLLMClientConfig


class MockResponse:

    def __init__(self, content: str):
        self.message = MockMessage(content)


class MockMessage:

    def __init__(self, content: str):
        self.content = content


class MockClient(BaseClient):

    def __init__(
        self,
        config: BaseLLMClientConfig,
        responses_dir: str,
    ) -> None:

        super().__init__(config=config)
        self.responses_dir = responses_dir
        self.call_count = 0
        self.call_history: list[dict[str, Any]] = []

        self._load_responses()

    def _load_responses(self) -> None:
        """Load all response files from the responses directory."""
        self.responses: list[str] = []
        i = 0
        while True:
            filepath = os.path.join(self.responses_dir, f"{i}.txt")
            if not os.path.exists(filepath):
                break
            with open(filepath, "r") as f:
                self.responses.append(f.read())
            i += 1

        if not self.responses:
            raise ValueError(f"No response files found in {self.responses_dir}")

    def _chat_completion_api(
        self, messages: list[dict], temperature: float, n: int = 1
    ) -> list:
        """Return pre-recorded responses instead of making real API calls."""
        if n != 1:
            raise NotImplementedError("MockClient only supports n=1")

        # Log the call for debugging/verification
        self.call_history.append(
            {
                "call_index": self.call_count,
                "messages": messages,
                "temperature": temperature,
                "n": n,
            }
        )

        # Check if we have more responses available
        if self.call_count >= len(self.responses):
            raise IndexError(
                f"No more responses available. Call count: {self.call_count}, "
                f"Available responses: {len(self.responses)}"
            )

        # Get the next response
        response_content = self.responses[self.call_count]
        self.call_count += 1

        # Return mock response in the expected format
        return [MockResponse(response_content)]

    def reset(self) -> None:
        """Reset call counter and history for reuse."""
        self.call_count = 0
        self.call_history = []
