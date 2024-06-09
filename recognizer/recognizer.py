from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import GenerationConfig, TrOCRProcessor, VisionEncoderDecoderModel
from transformers.utils import ModelOutput

from . import constants
from .output import RecognizerOutputs
from .smart_batcher import SmartBatcher


class Recognizer:
    """
    Recognizer is a wrapper class for the VisionEncoderDecoderModel to perform text recognition.

    This class provide an interface to deal with complex processes such as setting up the model, handle different input types,
    perform batch inference, clean text output, apply accelerator, and etc.

    Attributes
    ----------
        - model (VisionEncoderDecoderModel): The model to be used for inference.
        - processor (TrOCRProcessor): The processor to preprocess the images.
        - tokenizer (PreTrainedTokenizer): The tokenizer to tokenize the texts.

    Methods
    -------
        - load_recognizer: Loads a Recognizer from the given model_dir.
        - recognize: Recognizes text from a sequence of image paths, numpy arrays, or PIL images.

    Example
    -------
        >>> from looloo_ocr.recognizer import Recognizer
        >>> recognizer = Recognizer.load_recognizer(model_dir="path/to/model_dir")
        >>> inputs = ["path/to/image_1", "path/to/image_2"]
        >>> outputs = recognizer.recognize(inputs)
        >>> outputs.texts
        ["hello word", "look at me"]
        >>> outputs.token_ids
        [[78, 230], [23, 57, 908]]
        >>> outputs.scores
        [[0.98, 0.70], [0.99, 0.98, 0.97]]
        >>> outputs.low_confidence_texts
        [[(1, 'word', 0.70)], []]

    """

    def __init__(
        self,
        model: VisionEncoderDecoderModel,
        processor: TrOCRProcessor,
        device: str = constants.DEFAULT_DEVICE if torch.cuda.is_available() else "cpu",
    ) -> None:
        """
        Argument
        --------
            - model (VisionEncoderDecoderModel): The model to be used for inference.
            - processor (TrOCRProcessor): The processor to preprocess the images.
            - device (str, optional): The device to be used to run the Recognizer. Defaults to "cuda:0" if available else "cpu".
        """
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.device = device

        self._eos_token_id = self.tokenizer.eos_token_id

        # Set the model to eval mode.
        self.model.eval()

        # Convert model to half precision (fp16) for faster inference.
        self.model.to(device)
        if "cuda" in device:
            self.model.half()

    @classmethod
    def load_recognizer(
        cls,
        model_dir: str,
        device: str = constants.DEFAULT_DEVICE if torch.cuda.is_available() else "cpu",
    ) -> "Recognizer":
        """
        Loads a Recognizer from the given model_dir.

        Arguments
        ---------
            - model_dir (str): The path to the model to load.
            - device (str, optional): The device to be used to run the Recognizer. Defaults to "cuda:0" if available else "cpu".

        Returns
        -------
            Recognizer: a Recognizer object.

        Example
        -------
            >>> from looloo_ocr.recognizer import Recognizer
            >>> recognizer = Recognizer.load_recognizer(model_dir="path/to/model_dir")
        """
        # Load model from the given model_dir.
        model = VisionEncoderDecoderModel.from_pretrained(model_dir)
        # Load processor from the given model_dir.
        processor = TrOCRProcessor.from_pretrained(model_dir)

        # Create a recognizer object.
        recognizer = cls(
            model=model,
            processor=processor,
            device=device,
        )

        return recognizer

    def recognize(
        self,
        inputs: Sequence[str | np.ndarray | Image.Image],
        batch_size: int = constants.RECOGNIZER_BATCH_SIZE,
        smart_batching: bool = constants.RECOGNIZER_USE_SMART_BATCHING,
        confidence_threshold: float = constants.RECOGNIZER_CONFIDENCE_THRESHOLD,
        generation_config: Optional[GenerationConfig] = None,
        verbose: bool = True,
    ) -> RecognizerOutputs:
        """
        Recognizes text from a sequence of image paths, numpy arrays, or PIL images.

        Arguments
        ---------
            inputs (Sequence[str | np.ndarray | Image.Image]):
                A sequence of image paths, numpy arrays, or PIL images.
            batch_size (int, optional): Defaults to 16.
                The batch size to be used for inference. For CPU, batch_size=1 is recommended.
            smart_batching (bool, optional): Defaults to True.
                This will sort the images by width to reduce the padding size in each batch, hence reducing inference time.
            confidence_threshold (float, optional): Defaults to 0.9.
                The confidence threshold to justify as high confidence text.
            generation_config (Optional[GenerationConfig], optional): Defaults to None.
                The generation config to be used for inference.
            verbose (bool, optional): Defaults to True.
                Whether to print the progress bar.

        Returns
        -------
            RecognizerOutput: A RecognizerOutput object that contains the following attributes:

            - `predictions`: List of recognized texts.
            - `token_ids`: List of token IDs corresponding to the recognized texts.
            - `scores`: List of confidence scores for the recognized texts.
            - `low_confidence_texts`: List of low confidence texts.

        Example
        -------
            >>> from looloo_ocr.recognizer import Recognizer
            >>> recognizer = Recognizer.load_recognizer(model_dir="path/to/model_dir")
            >>> inputs = ["path/to/image_1", "path/to/image_2"]
            >>> outputs = recognizer.recognize(inputs)
            >>> outputs.predictions
            ["hello word", "look at me"]
            >>> outputs.token_ids
            [[78, 230], [23, 57, 908]]
            >>> outputs.scores
            [[0.98, 0.70], [0.99, 0.98, 0.97]]
            >>> outputs.low_confidence_texts
            [[(1, 'word', 0.70)], []]
        """
        # Check if all inputs are of the same type.
        if not isinstance(inputs, Sequence):
            raise TypeError(f"Inputs must be a sequence, but got: {type(inputs)}")

        all_input_types = set(type(input) for input in inputs)
        if len(all_input_types) > 1:
            raise TypeError(f"All inputs must be of the same type. Got: {all_input_types}")

        # Check if the input type is supported.
        if any(input_type not in constants.RECOGNIZER_SUPPORTED_TYPES for input_type in all_input_types):
            raise TypeError(
                f"Input type not supported. Supported input types: {constants.RECOGNIZER_SUPPORTED_TYPES}, but got: {all_input_types}"
            )

        # Set generation config
        if generation_config is None:
            generation_config = constants.RECOGNIZER_GENERATION_CONFIG

        # Get the input type.
        input_sample = inputs[0]

        # Sort the image list beforehand if smart batching is enabled.
        do_smart_batching = False
        if smart_batching:
            # Check if smart batching is supported for the given input type.
            do_smart_batching = SmartBatcher.is_supported(input_sample, verbose=verbose)

            # Sort the images by width if smart batching is enabled.
            if do_smart_batching:
                inputs, sort_orders = SmartBatcher.sort_items(inputs)

        # Split the inputs into batches.
        image_batches = SmartBatcher.split_sequence_to_batches(inputs, batch_size)

        # Iterate each batch and predict the outputs.
        batch_outputs = []
        for image_batch in tqdm(image_batches, total=len(image_batches), disable=not verbose):
            # Load the images if the input is an image paths.
            if isinstance(input_sample, str):
                image_batch = [Image.open(image_path).convert("RGB") for image_path in image_batch]

            # Predict the outputs for the current batch.
            batch_output = self._predict(
                images=image_batch,
                confidence_threshold=confidence_threshold,
                generation_config=generation_config,
            )
            # Append the batch_output to the outputs.
            batch_outputs.append(batch_output)

        # Join all the outputs into a single RecognizerOutput.
        output = RecognizerOutputs.join(batch_outputs)

        # Sort the outputs back to the original order if smart batching is performed.
        if do_smart_batching:
            output.texts = SmartBatcher.reorder_items(output.texts, sort_orders)
            output.token_ids = SmartBatcher.reorder_items(output.token_ids, sort_orders)
            output.scores = SmartBatcher.reorder_items(output.scores, sort_orders)
            output.low_confidence_texts = SmartBatcher.reorder_items(output.low_confidence_texts, sort_orders)

        return output

    @torch.no_grad()
    def _predict(
        self,
        images: list[Image.Image],
        generation_config: GenerationConfig,
        confidence_threshold: float = 0.9,
    ) -> RecognizerOutputs:
        """Predict images and returns the RecognizerOutput."""
        # Convert the input images to tensor and assign to the device.
        pixel_values = self.processor(images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        # Predict images.
        batch_output = self.model.generate(
            pixel_values,
            output_scores=["scores"],
            return_dict_in_generate=True,
            generation_config=generation_config,
        )

        # Get predicted sequences.
        sequences = batch_output.sequences

        # Decode and clean token_ids to get the predicted texts.
        texts = self.processor.tokenizer.batch_decode(
            sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        # texts = [Normalizer.normalize(text) for text in texts]

        # Get the confidence scores for every sequence in the batch.
        token_ids, confidences_scores = self._extract_token_ids_and_confidence_scores(
            batch_output, generation_config.num_beams
        )

        # Get the low confidence scores.
        low_confidence_texts = self._extract_low_confidence_scores(sequences, confidences_scores, confidence_threshold)

        # Construct the RecognizerOutput.
        output = RecognizerOutputs(
            texts=texts,
            token_ids=token_ids,
            scores=confidences_scores,
            low_confidence_texts=low_confidence_texts,
        )

        return output

    def _extract_token_ids_and_confidence_scores(
        self, model_output: ModelOutput, num_beams: int
    ) -> tuple[list[list[int]], list[list[float]]]:
        """
        Extract confidence scores from the model output.
        As described in this [link](https://huggingface.co/docs/transformers/main_classes/output#transformers.utils.ModelOutput).
        """
        # Check if the model_output is an instance of ModelOutput.
        if not isinstance(model_output, ModelOutput):
            raise TypeError(f"Input must be an instance of ModelOutput, but got: {type(model_output)}")

        # Get the token ids and their confidence scores.
        token_ids = []
        confidences_scores = []
        total_sequences = len(model_output.sequences)

        # Iterate each sequence in the batch.
        for sequence_idx in range(total_sequences):
            # Get the confidence scores for each token in the sequence.
            sequence_token_ids = []
            sequence_confidence_score_list = []

            # Iterate each token in the sequence.
            for score in model_output.scores:
                best_confidence_index = num_beams * sequence_idx
                token_score = score[best_confidence_index]
                token_probability = token_score.softmax(dim=-1)
                token = token_probability.argmax(dim=-1).item()

                # If reach the end of the sequence, stop.
                if token == self._eos_token_id:
                    break

                confidence = token_probability[token].cpu().item()

                sequence_token_ids.append(token)
                sequence_confidence_score_list.append(confidence)

            token_ids.append(sequence_token_ids)
            confidences_scores.append(sequence_confidence_score_list)

        return token_ids, confidences_scores

    def _extract_low_confidence_scores(
        self,
        token_ids_sequence: Sequence[Sequence[int]],
        confidences_scores_sequence: Sequence[Sequence[float]],
        confidence_threshold: float,
    ) -> list[tuple[int, str, float]]:
        """Extract low confidence scores from sequences and their confidence scores."""
        low_confidence_texts_sequence = []

        for token_ids, confidence_scores in zip(token_ids_sequence, confidences_scores_sequence):
            low_confidence_tokens = []
            token_texts = [self.tokenizer.decode(_id) for _id in token_ids]

            for token_idx, (token_text, confidence_score) in enumerate(zip(token_texts, confidence_scores)):
                if confidence_score < confidence_threshold:
                    low_confidence_tokens.append((token_idx, token_text, confidence_score))

            low_confidence_texts_sequence.append(low_confidence_tokens)

        return low_confidence_texts_sequence
