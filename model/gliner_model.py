import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class GLiNER(nn.Module):
    def __init__(self, pretrained_model_name="microsoft/deberta-v3-base", span_max_length=2, hidden_size=768, dropout_rate=0.4):
        super(GLiNER, self).__init__()
        # Initialize the tokenizer and the pretrained model
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)

        # Output size of the encoder model
        self.encoder_output_size = self.encoder.config.hidden_size
        
        # Feed-Forward Network (FFN) to refine entity embeddings
        self.entity_ffn = nn.Sequential(
            nn.Linear(self.encoder_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # FFN for span embeddings
        self.span_ffn = nn.Sequential(
            nn.Linear(2 * self.encoder_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Sigmoid activation for final scores
        self.sigmoid = nn.Sigmoid()
        
        # Maximum span length
        self.span_max_length = span_max_length

        # Weight for positive classes in the loss function
        pos_weight = torch.tensor([5], dtype=torch.float32)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


    def forward(self, input_ids, attention_masks, entity_types, spans, sentence_masks, entity_masks, binary_labels=None):
        # Pass the input data through the encoder model
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_masks)
        token_embeddings = outputs.last_hidden_state
    
        # Separate embeddings into entities and text
        entity_embeddings, text_embeddings = self.split_embeddings(token_embeddings, len(entity_types[0]))
        
        # Refine entity embeddings
        refined_entity_embeddings = self.entity_ffn(entity_embeddings)
        
        # Compute scores for spans
        span_scores, scores_logit = self.compute_span_scores(refined_entity_embeddings, text_embeddings, spans)

        # If binary labels are provided, compute the loss
        if binary_labels is not None:
            loss = self.compute_loss(scores_logit, binary_labels)
            return span_scores, loss
        
        return span_scores


    def split_embeddings(self, token_embeddings, num_entity_types=25):
        # Extract entity embeddings and text embeddings from encoder output
        entity_embeddings = token_embeddings[:, 0:num_entity_types, :]
        text_embeddings = token_embeddings[:, num_entity_types + 1:, :]
        
        return entity_embeddings, text_embeddings

  
    def compute_span_scores(self, entity_embeddings, text_embeddings, spans):
        """
        Compute span scores in a vectorized manner.
        """
        batch_size, text_length, hidden_size = text_embeddings.shape

        # Convert spans into a tensor suitable for PyTorch
        spans_tensor = torch.stack([s.clone().detach().to(device=text_embeddings.device) for s in spans])

        # Get embeddings for the start and end positions of spans
        i_indices = spans_tensor[:, :, 0].unsqueeze(-1).expand(-1, -1, hidden_size)
        j_indices = spans_tensor[:, :, 1].unsqueeze(-1).expand(-1, -1, hidden_size)

        start_embeddings = torch.gather(text_embeddings, 1, i_indices)  # Embeddings for start indices
        end_embeddings = torch.gather(text_embeddings, 1, j_indices)    # Embeddings for end indices

        # Concatenate start and end embeddings, then pass through the FFN
        span_reprs = torch.cat([start_embeddings, end_embeddings], dim=-1)  # Merge start-end embeddings
        span_reprs = self.span_ffn(span_reprs)                              # Transform through FFN

        # Compute span scores relative to entity embeddings
        scores = torch.einsum("bsh,beh->bse", span_reprs, entity_embeddings)  # Tensor product for scores

        # Apply Sigmoid activation to final scores
        span_scores = self.sigmoid(scores)

        return span_scores, scores


    def compute_loss(self, span_scores, binary_labels):
        """
        Compute binary cross-entropy loss with weights for the positive class.
        """
        # BCEWithLogitsLoss expects raw scores (without Sigmoid), so self.sigmoid is excluded here.
        loss = self.loss_fn(span_scores, binary_labels)
        return loss
    
def load_model(model_path="model_fullNewBCE26.pth"):
    return torch.load(model_path)