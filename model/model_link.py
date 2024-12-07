import string
import torch
from importlib import reload
from model import gliner_model

reload(gliner_model)
from model.gliner_model import GLiNER

MAX_ENTITY_PER_SEQ = 10
MAX_LENGTH = 213
MAX_SPAN_LENGTH = 4


def tokenize_sentence_with_entities(sentence, entity_types_to_detect, model):
    """
    Tokenizes a sentence while detecting and incorporating entity types.

    Args:
        sentence (str): The input sentence to process.
        entity_types_to_detect (list): A list of entity types to detect (e.g., ["person", "organization"]).
        model: The model containing the tokenizer to use.

    Returns:
        tuple: Contains tokenized sentence, reconstructed tokens, first subtoken IDs, and entity tokens.
    """
    current_entity_ids = []
    current_entity_strs = []

    for entity_type in entity_types_to_detect:
        entity_token = f"[ENT] {entity_type}"
        entity_token_id = model.tokenizer.convert_tokens_to_ids(entity_token)

        # Handle tokenization issues
        if isinstance(entity_token_id, int):
            if entity_token_id not in current_entity_ids:
                current_entity_ids.append(entity_token_id)
                current_entity_strs.append(entity_type)
        else:
            print(f"Entity '{entity_type}' is not recognized in any form by the tokenizer.")

    entity_tokens = " ".join(f"[ENT] {et}" for et in current_entity_strs)

    # Tokenize the sentence
    encoded = model.tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, add_special_tokens=False)
    input_ids = encoded["input_ids"][0]
    sentence_tokens = model.tokenizer.convert_ids_to_tokens(input_ids)

    reconstructed_tokens, first_subtoken_ids = reconstruct_tokens(sentence_tokens, input_ids)

    return sentence_tokens, reconstructed_tokens, first_subtoken_ids, entity_tokens, current_entity_strs, current_entity_ids


def reconstruct_tokens(sentence_tokens, input_ids):
    """
    Reconstructs words from sentence tokens and extracts first subtoken IDs.

    Args:
        sentence_tokens (list): List of tokenized sentence.
        input_ids (list): Corresponding input IDs for the tokens.

    Returns:
        tuple: Reconstructed tokens and list of first subtoken IDs.
    """
    reconstructed_tokens = []
    first_subtoken_ids = []
    current_word = ""
    current_word_has_id = False  # To track if the current word already has an ID

    for i, token in enumerate(sentence_tokens):
        # New word starts
        if token.startswith("▁"):
            # Save the previous word if it exists
            if current_word:
                reconstructed_tokens.append(current_word)
                if not current_word_has_id:
                    first_subtoken_ids.append(input_ids[i - 1].item())
            # Start a new word
            current_word = token[1:]
            first_subtoken_ids.append(input_ids[i].item())
            current_word_has_id = True
        # Punctuation (considered a new "word")
        elif token in string.punctuation:
            if current_word:  # Save the current word before punctuation
                reconstructed_tokens.append(current_word)
                if not current_word_has_id:
                    first_subtoken_ids.append(input_ids[i - 1].item())
            current_word = ""
            reconstructed_tokens.append(token)
            first_subtoken_ids.append(input_ids[i].item())
            current_word_has_id = False
        # Continuation of the current word
        else:
            current_word += token
            current_word_has_id = current_word_has_id or (len(first_subtoken_ids) > len(reconstructed_tokens))

    # Add the last word if it exists
    if current_word:
        reconstructed_tokens.append(current_word)
        if not current_word_has_id:
            first_subtoken_ids.append(input_ids[-1].item())

    # Validation step
    if len(reconstructed_tokens) != len(first_subtoken_ids):
        raise ValueError(
            f"Mismatch: {len(reconstructed_tokens)} reconstructed tokens vs {len(first_subtoken_ids)} first subtoken IDs"
        )

    return reconstructed_tokens, first_subtoken_ids


def prepare_input_for_model(entity_tokens, current_entity_strs, current_entity_ids, first_subtoken_ids, model):
    """
    Prepares input data for the model by encoding entity tokens and constructing necessary tensors.

    Args:
        entity_tokens (str): Entity tokens.
        current_entity_strs (list): Entity strings.
        current_entity_ids (list): Entity IDs.
        first_subtoken_ids (list): First subtoken IDs.
        model: The model containing the tokenizer to use.

    Returns:
        tuple: Tensors for entities, input IDs, attention mask, entity mask, and sentence mask.
    """
    encoded_entity = model.tokenizer(entity_tokens, return_tensors="pt", padding="max_length", truncation=True, add_special_tokens=False)
    encoded_entity = encoded_entity["input_ids"][0].tolist() + [0] * (MAX_ENTITY_PER_SEQ - len(current_entity_strs))

    sep_id = model.tokenizer.convert_tokens_to_ids('[SEP]')
    combined_ids = encoded_entity + [sep_id] + first_subtoken_ids
    combined_ids = combined_ids[:MAX_LENGTH] + [0] * (MAX_LENGTH - len(combined_ids))

    attention_mask = [1 if id != 0 else 0 for id in combined_ids]
    entity_mask = [1 if i < len(current_entity_strs) else 0 for i in range(len(combined_ids))]
    sentence_mask = [1 if i > len(encoded_entity) and combined_ids[i] != 0 and combined_ids[i] != sep_id else 0 for i in range(len(combined_ids))]

    current_entity_ids = current_entity_ids + [0] * (MAX_ENTITY_PER_SEQ - len(current_entity_strs))

    entity_tensor = torch.tensor(current_entity_ids, dtype=torch.long)
    input_ids_tensor = torch.tensor(combined_ids, dtype=torch.long)
    attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
    entity_mask_tensor = torch.tensor(entity_mask, dtype=torch.long)
    sentence_mask_tensor = torch.tensor(sentence_mask, dtype=torch.long)

    return entity_tensor, input_ids_tensor, attention_mask_tensor, entity_mask_tensor, sentence_mask_tensor


def create_spans(input_ids_tensor, entity_tensor):
    """
    Creates span pairs for the sentence based on the input and entity tensors.

    Args:
        input_ids_tensor (Tensor): Input IDs tensor.
        entity_tensor (Tensor): Entity IDs tensor.

    Returns:
        tuple: List of spans and corresponding tensor of spans.
    """
    num_tokens = len(input_ids_tensor) - len(entity_tensor) - 1
    spans = [(start, end) for start in range(num_tokens) for end in range(start, min(start + MAX_SPAN_LENGTH, num_tokens))]
    spans_tensor = torch.tensor(spans, dtype=torch.long)
    return spans, spans_tensor


def evaluate_model(model, input_ids_tensor, attention_mask_tensor, entity_tensor, spans_tensor, sentence_mask_tensor, entity_mask_tensor):
    """
    Evaluates the model on the provided tensors.

    Args:
        model: The model to evaluate.
        input_ids_tensor (Tensor): Input IDs tensor.
        attention_mask_tensor (Tensor): Attention mask tensor.
        entity_tensor (Tensor): Entity tensor.
        spans_tensor (Tensor): Spans tensor.
        sentence_mask_tensor (Tensor): Sentence mask tensor.
        entity_mask_tensor (Tensor): Entity mask tensor.

    Returns:
        Tensor: Span scores from the model.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        span_scores = model(
            input_ids=input_ids_tensor.unsqueeze(0).to(device),
            attention_masks=attention_mask_tensor.unsqueeze(0).to(device),
            entity_types=entity_tensor.unsqueeze(0).to(device),
            spans=spans_tensor.unsqueeze(0).to(device),
            sentence_masks=sentence_mask_tensor.unsqueeze(0).to(device),
            entity_masks=entity_mask_tensor.unsqueeze(0).to(device)
        )
    return span_scores


def threshold_span_scores(span_scores, threshold_score):
    """
    Converts span scores into binary values based on a threshold.

    Args:
        span_scores (Tensor): Span scores from the model.
        threshold_score (float): The threshold score to classify spans.

    Returns:
        Tensor: Binary span scores.
    """
    return (span_scores > threshold_score).int()


def max_mask_span_scores(span_scores, binary_span_scores):
    """
    Applies a mask to select the spans with the highest score for each entity.
    
    This function creates a mask where the highest scoring span for each entity
    in the span_scores tensor is set to 1. It then applies this mask to the 
    binary span scores to retain only the most important spans.

    Args:
        span_scores (Tensor): Span scores from the model, shape [batch_size, num_spans, num_entities].
        binary_span_scores (Tensor): Binary span scores, shape [batch_size, num_spans, num_entities].

    Returns:
        Tensor: Masked binary span scores with the highest span score selected for each entity.
    """
    # Create a mask with 0s and 1s representing the highest value in each list
    # Initialize a mask with the same dimensions as span_scores
    max_mask = torch.zeros_like(span_scores, dtype=torch.int)
    
    # Iterate through each batch, span, and entity to identify the index of the maximum values
    for i in range(span_scores.size(1)):  # Iterate over the spans dimension
        for j in range(span_scores.size(2)):  # Iterate over the entities dimension
            # Find the index of the maximum value in the list for this span and entity
            max_index = torch.argmax(span_scores[0, i, :])  # Corrected for dimensions
            # Set the index of the max value to 1 in the mask
            max_mask[0, i, max_index] = 1  # Corrected for dimensions

    # Apply the mask on the binary_span_scores
    masked_binary_span_scores = binary_span_scores * max_mask

    return masked_binary_span_scores


def get_spans_result(spans, span_scores, binary_span_scores_list, first_subtoken_ids, entity_types_to_detect):
    """
    Extracts and processes the spans with detected entities.

    Args:
        spans (list): List of spans.
        span_scores (Tensor): Model span scores.
        binary_span_scores_list (list): Binary span scores.
        first_subtoken_ids (list): List of first subtoken IDs.
        entity_types_to_detect (list): List of entity types to detect.

    Returns:
        list: Spans with detected entities.
    """
    spans_result = []
    max_index = len(first_subtoken_ids)

    for i, example in enumerate(binary_span_scores_list):
        for j, span_scores in enumerate(example):
            associated_span = spans[j]
            if associated_span[0] < max_index and associated_span[1] < max_index:
                for k, score in enumerate(span_scores[:len(entity_types_to_detect)]):
                    if score == 1:
                        spans_result.append((associated_span, entity_types_to_detect[k]))

    print("Spans with detected entities:")
    for span, entity_type in spans_result:
        print(f"{span} -> {entity_type}")

    return spans_result

def clean_spans_nested(spans_result):
    """
    Cleans the nested spans, keeping only the largest ones.

    Args:
        spans_result (list): List of spans with detected entities.

    Returns:
        list: Cleaned spans with detected entities, keeping only the largest ones.
    """
    cleaned_spans = []
    
    # Sort spans by their start index, and if they are the same, by their end index in descending order
    spans_result.sort(key=lambda x: (x[0][0], -x[0][1]))
    
    for current_span, current_entity_type in spans_result:
        # Check if the current span is contained in any already added span
        contained = False
        for saved_span, _ in cleaned_spans:
            # If current_span is contained in saved_span, skip it
            if saved_span[0] <= current_span[0] and saved_span[1] >= current_span[1]:
                contained = True
                break
        
        # If it's not contained in any span, add it
        if not contained:
            cleaned_spans.append((current_span, current_entity_type))
    
    # Print cleaned spans
    print("Cleaned spans with detected entities:")
    for span, entity_type in cleaned_spans:
        print(f"{span} -> {entity_type}")
    
    return cleaned_spans

def spans_to_text(reconstructed_tokens, spans_result):
    """
    Converts detected spans to a formatted text with entity annotations.

    Args:
        reconstructed_tokens (list): Reconstructed tokens of the sentence.
        spans_result (list): List of spans with detected entities.

    Returns:
        str: Formatted sentence with entity annotations.
    """
    sentence_output_str = ""
    last_ind = -1
    for index, token in enumerate(reconstructed_tokens):
        span_detected = False
        for span, entity_type in spans_result:
            if span[0] == index:
                sentence_output_str += '{' + ' '.join(reconstructed_tokens[span[0]:span[1] + 1]) + '} [' + entity_type + '] '
                span_detected = True
                last_ind = span[1]
        if not span_detected and last_ind < index:
            last_ind = -1
            sentence_output_str += token + ' '

    print(f'Result : {sentence_output_str}')
    return sentence_output_str


class EntityDetectionModel:
    def __init__(self, model_path="model_fullNewBCE31.pth"):
        """
        Initializes and loads the model from a given path.

        Args:
            model_path (str): Path to the model file.
        """
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        """
        Loads the model into memory.
        """
        if self.model is None:
            print("Loading model into memory...")
            self.model = gliner_model.load_model(self.model_path)
            print("Model loaded successfully.")
        else:
            print("Model already loaded.")

    def process(self, sentence, entity_types, threshold_score,nested_ner):
        """
        Processes a sentence to detect entities based on provided entity types and threshold.

        Args:
            sentence (str): Sentence to analyze.
            entity_types (str): Comma-separated entity types to detect.
            threshold_score (float): Threshold score for entity detection.

        Returns:
            str: Annotated sentence with detected entities.
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Call load_model() before processing sentences.")

        # Print the sentence and the entity types to detect for debugging
        print(f"Processing sentence: {sentence}")
        print(f"Entity types to detect: {entity_types}")

        entity_types_to_detect = [item.lower() for item in entity_types.split(', ')]

        (sentence_tokens, reconstructed_tokens, first_subtoken_ids, entity_tokens, 
         current_entity_strs, current_entity_ids) = tokenize_sentence_with_entities(sentence, entity_types_to_detect, self.model)

        (entity_tensor, input_ids_tensor, attention_mask_tensor, entity_mask_tensor, sentence_mask_tensor) = prepare_input_for_model(
            entity_tokens, current_entity_strs, current_entity_ids, first_subtoken_ids, self.model
        )

        spans, spans_tensor = create_spans(input_ids_tensor, entity_tensor)

        span_scores = evaluate_model(self.model, input_ids_tensor, attention_mask_tensor, entity_tensor, 
                                     spans_tensor, sentence_mask_tensor, entity_mask_tensor)

        binary_span_scores = threshold_span_scores(span_scores, threshold_score)
        masked_binary_span_scores = max_mask_span_scores(span_scores, binary_span_scores)
        binary_span_scores_list = masked_binary_span_scores.cpu().numpy().tolist()

        spans_result = get_spans_result(spans, span_scores, binary_span_scores_list, first_subtoken_ids, entity_types_to_detect)
        
        if not nested_ner:
            spans_result = clean_spans_nested(spans_result)

        sentence_output_str = spans_to_text(reconstructed_tokens, spans_result)

        return sentence_output_str


