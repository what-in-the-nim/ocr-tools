from dataclasses import dataclass
from typing import Iterator, Sequence


@dataclass(slots=True)
class RecognizerOutput:
    """
    RecognizerOutput is a class that holds a single prediction from the Recognizer.

    Attributes:
    ----------
        text (str): The predicted text.
        token_ids (Sequence[int]): A sequence of token ids.
        score (float): The minimum score of the prediction.
        token_scores (Sequence[float]): A sequence of scores.
        low_confidence_text (tuple[int, str, float]): A tuple containing the index, text, and score of the low confidence text.
    """

    text: str
    token_ids: Sequence[int]
    token_scores: Sequence[float]
    low_confidence_text: tuple[int, str, float]

    @property
    def min_score(self) -> float:
        if len(self.token_scores) == 0:
            return 0.0
        return min(self.token_scores)

    @property
    def average_score(self) -> float:
        if len(self.token_scores) == 0:
            return 0.0
        return sum(self.token_scores) / len(self.token_scores)


@dataclass
class RecognizerOutputs:
    """
    RecognizerOutputs is a class that holds the output of the Recognizer prediction.

    Attributes:
    ----------
        texts (Sequence[str]): A sequence of predicted texts.
        token_ids (Sequence[Sequence[int]]): A sequence of token ids.
        scores (Sequence[Sequence[float]]): A sequence of scores.
        low_confidence_texts (Sequence[tuple[int, str, float]]): A sequence of low confidence texts.

    Examples:
    --------
    You can access a RecognizerOutputs instance using the index operator.
        >>> output = recognizer.recognize(inputs)
        >>> output.texts
        ["Hello, it's my", "Mario and Luigi"]
        >>> output.token_ids
        [[10, 92, 43, 64, 58], [64, 43, 92, 10, 58]]
        >>> output.scores
        [[0.99, 0.97, 0.96, 0.95, 0.89], [0.99, 0.97, 0.96, 0.95, 0.94]]
        >>> output.low_confidence_texts
        [(4, "my", 0.89), ()]

    You can also iterate over a RecognizerOutputs instance to get the output at each index.
        >>> for prediction in output:
        ...     print(f"Text: {prediction.text}")
        ...     print(f"Token IDs: {prediction.token_ids}")
        ...     print(f"Scores: {prediction.score}")
        ...     print(f"Token Scores: {prediction.token_scores}")
        ...     print(f"Low Confidence Text: {prediction.low_confidence_text}")
        Text: Hello, it's my
        Token IDs: [10, 92, 43, 64, 58]
        Scores: 0.89
        Token Scores: [0.99, 0.97, 0.96, 0.95, 0.89]
        Low Confidence Text: (4, "my", 0.89)

    You can also concatenate two RecognizerOutputs instances using the add operator.
        >>> output_1 = recognizer.recognize(inputs_1)
        >>> output_2 = recognizer.recognize(inputs_2)
        >>> output_1 + output_2
        RecognizerOutputs(total_prediction=4)

    You can also concatenate a list of RecognizerOutputs instances using the join class method.
        >>> outputs = [output_1, output_2, ...]
        >>> joined_output = RecognizerOutputs.join(outputs)
    """

    texts: Sequence[str]
    token_ids: Sequence[Sequence[int]]
    scores: Sequence[Sequence[float]]
    low_confidence_texts: Sequence[tuple[int, str, float]]

    def __post_init__(self) -> None:
        """Check if all the inputs have the same length."""
        # Find the length of each input
        total_texts = len(self.texts)
        total_token_ids = len(self.token_ids)
        total_scores = len(self.scores)
        total_low_confidence_texts = len(self.low_confidence_texts)

        # Check if all the inputs have the same length
        if len(set([total_texts, total_token_ids, total_scores, total_low_confidence_texts])) != 1:
            raise ValueError(
                f"Length of all inputs must be the same. "
                f"Got {total_texts} texts, {total_token_ids} token ids, {total_scores} scores, "
                f"and {total_low_confidence_texts} low confidence texts."
            )

    def __len__(self) -> int:
        """Return the number of output inside the class."""
        return len(self.texts)

    def __getitem__(self, index: int) -> RecognizerOutput:
        """Return the output at the given index."""
        return RecognizerOutput(
            text=self.texts[index],
            token_ids=self.token_ids[index],
            token_scores=self.scores[index],
            low_confidence_text=self.low_confidence_texts[index],
        )

    def __iter__(self) -> Iterator[RecognizerOutput]:
        """Return an interator for each output."""
        self.index = 0
        return self

    def __next__(self) -> RecognizerOutput:
        # Ensure the index is within the bounds of the lists
        if self.index < len(self):
            # Get all the output at the current index
            item = self.__getitem__(self.index)
            self.index += 1
            return item
        else:
            # Reset index and raise StopIteration to signal the end of iteration
            self.index = 0
            raise StopIteration

    def __add__(self, other: "RecognizerOutputs") -> "RecognizerOutputs":
        """Concatenate two RecognizerOutputs instances."""
        combined_predictions = self.texts + other.texts
        combined_token_ids = self.token_ids + other.token_ids
        combined_scores = self.scores + other.scores
        combined_low_confidence_texts = self.low_confidence_texts + other.low_confidence_texts

        return RecognizerOutputs(
            texts=combined_predictions,
            token_ids=combined_token_ids,
            scores=combined_scores,
            low_confidence_texts=combined_low_confidence_texts,
        )

    def __str__(self) -> str:
        """Return a string representation of the class."""
        return f"RecognizerOutputs(total_prediction={len(self)})"

    def __repr__(self) -> str:
        """Return a string representation of the class."""
        return self.__str__()

    @classmethod
    def join(cls, outputs: Sequence["RecognizerOutputs"]) -> "RecognizerOutputs":
        """Concatenate a list of RecognizerOutputs instances."""
        combined_predictions = []
        combined_token_ids = []
        combined_scores = []
        combined_low_confidence_texts = []

        for output in outputs:
            combined_predictions.extend(output.texts)
            combined_token_ids.extend(output.token_ids)
            combined_scores.extend(output.scores)
            combined_low_confidence_texts.extend(output.low_confidence_texts)

        return cls(
            texts=combined_predictions,
            token_ids=combined_token_ids,
            scores=combined_scores,
            low_confidence_texts=combined_low_confidence_texts,
        )

    def remove(self, index: int | Sequence[int]) -> "RecognizerOutputs":
        """Remove the output at the given index."""
        # Check if index is valid
        if not isinstance(index, (int, Sequence)):
            raise TypeError(f"index must be an integer or a sequence of integers, got {type(index)}")
        # Convert index to a sequence if it is an integer
        if isinstance(index, int):
            index = [index]
        # Check if all index are integers
        if not all(isinstance(i, int) for i in index):
            raise TypeError(f"index must be an integer or a sequence of integers, got {type(index)}")
        # Check if all index are within the bounds of the lists
        max_index = max(index)
        if max_index >= len(self):
            raise IndexError(f"index {max_index} is out of bounds for RecognizerOutputs with length {len(self)}")

        # Remove the output at the given index
        texts = [text for i, text in enumerate(self.texts) if i not in index]
        token_ids = [token_id for i, token_id in enumerate(self.token_ids) if i not in index]
        scores = [score for i, score in enumerate(self.scores) if i not in index]
        low_confidence_texts = [
            low_confidence_text for i, low_confidence_text in enumerate(self.low_confidence_texts) if i not in index
        ]

        return RecognizerOutputs(
            texts=texts,
            token_ids=token_ids,
            scores=scores,
            low_confidence_texts=low_confidence_texts,
        )
