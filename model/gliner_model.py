import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class GLiNER(nn.Module):
    def __init__(self, pretrained_model_name="microsoft/deberta-v3-base", span_max_length=2, hidden_size=768, dropout_rate=0.4):
        super(GLiNER, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)

        self.encoder_output_size = self.encoder.config.hidden_size
        self.entity_ffn = nn.Sequential(
            nn.Linear(self.encoder_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.span_ffn = nn.Sequential(
            nn.Linear(2 * self.encoder_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.sigmoid = nn.Sigmoid()
        
        self.span_max_length = span_max_length
        # self.loss_fn = nn.BCELoss()  # Binary Cross-Entropy Loss

        pos_weight = torch.tensor([5], dtype=torch.float32)  # Convertir en tenseur si nécessaire
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, input_ids, attention_masks, entity_types, spans, sentence_masks, entity_masks, binary_labels=None):
        # print("Input IDs shape:", input_ids.shape)
        # print("Attention mask shape:", attention_masks.shape)
        # Passer input_ids et attention_masks au modèle
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_masks)
        token_embeddings = outputs.last_hidden_state
    
        entity_embeddings, text_embeddings = self.split_embeddings(token_embeddings,len(entity_types[0]))
        

        refined_entity_embeddings = self.entity_ffn(entity_embeddings)
        
        span_scores,scores_logit = self.compute_span_scores(refined_entity_embeddings, text_embeddings, spans)

        if binary_labels is not None:
            #loss = self.compute_loss(span_scores, binary_labels)
            loss = self.compute_loss(scores_logit,binary_labels)
            return span_scores, loss
        
        return span_scores


    def split_embeddings(self, token_embeddings, num_entity_types = 25):
        entity_embeddings = token_embeddings[:, 0:num_entity_types, :]
        text_embeddings = token_embeddings[:, num_entity_types + 1:, :]
        
        return entity_embeddings, text_embeddings

    
    def compute_span_scores(self, entity_embeddings, text_embeddings, spans):
        """
        Calcule les scores des spans en une seule passe vectorisée, 
        en supposant que tous les spans sont valides.
        """
        batch_size, text_length, hidden_size = text_embeddings.shape

        # Conversion des spans en tensor directement
        spans_tensor = torch.stack([torch.tensor(s, device=text_embeddings.device) for s in spans])  # (batch, num_spans, 2)

        # Récupération des embeddings des spans
        i_indices = spans_tensor[:, :, 0].unsqueeze(-1).expand(-1, -1, hidden_size)  # (batch, num_spans, hidden_size)
        j_indices = spans_tensor[:, :, 1].unsqueeze(-1).expand(-1, -1, hidden_size)

        start_embeddings = torch.gather(text_embeddings, 1, i_indices)  # (batch, num_spans, hidden_size)
        end_embeddings = torch.gather(text_embeddings, 1, j_indices)    # (batch, num_spans, hidden_size)

        # Concaténer les embeddings des extrémités et passer dans la FFN
        span_reprs = torch.cat([start_embeddings, end_embeddings], dim=-1)  # (batch, num_spans, 2 * hidden_size)
        span_reprs = self.span_ffn(span_reprs)                              # (batch, num_spans, hidden_size)

        # Calcul des scores pour toutes les entités
        scores = torch.einsum("bsh,beh->bse", span_reprs, entity_embeddings)  # (batch, num_spans, num_entity_types)

        # Appliquer la sigmoïde pour les scores finaux
        span_scores = self.sigmoid(scores)

        return span_scores,scores


    def compute_loss(self, span_scores, binary_labels):
        """
        Calcul de la perte binaire cross-entropy entre les scores et les étiquettes.
        """
        # print(f"span_scores shape: {span_scores.shape}")
        # print(f"binary_labels shape: {binary_labels.shape}")

        # Appliquer la perte
        loss = self.loss_fn(span_scores, binary_labels)
        return loss

    def compute_loss(self, span_scores, binary_labels):
        """
        Calcul de la perte binaire cross-entropy avec des poids pour la classe positive.
        """
        # BCEWithLogitsLoss attend des scores bruts (sans Sigmoid), donc on peut retirer self.sigmoid ici.
        loss = self.loss_fn(span_scores, binary_labels)
        return loss
    
def load_model(model_path="model_fullNewBCE26.pth"):
    return torch.load(model_path)

