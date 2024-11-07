# Entity-centered-Cross-document-Relation-Extraction

This project implements the method presented in the paper "Entity-centered Cross-document Relation Extraction" by Fengqi Wang et al., which addresses the task of extracting relationships between entities across multiple documents. Unlike traditional methods that focus on extracting relations within a single document, this approach tackles the challenge of cross-document relation extraction (RE) by leveraging entity-centered information and contextual relationships between text paths in different documents.

## Key Features
- Entity-based Document Context Filtering
The method filters out noisy and irrelevant sentences by retaining only the useful information that connects the target entities across multiple documents. This is done using bridge entities that link text paths within the documents.

- Cross-path Entity Relation Attention
A novel attention mechanism is introduced to allow entity relations across different text paths to interact with each other, improving the overall extraction process.

- State-of-the-Art Performance
The method is evaluated on the CodRED dataset, demonstrating a significant performance improvement, with an F1-score improvement of at least 10% compared to existing methods.
