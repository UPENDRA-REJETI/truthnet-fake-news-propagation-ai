# Experiment 01 — GraphSAGE on UPFD GossipCop

## Objective

To develop a Graph Neural Network baseline for fake-news classification using propagation graphs from the UPFD GossipCop dataset.

## Dataset

Dataset: UPFD GossipCop

Feature type: profile

Train graphs: 1092

Validation graphs: 546

Test graphs: 3826

Node features: 10

Classes: 2

## Input

Each sample is a propagation graph.

- Nodes represent users/entities in the propagation graph.
- Edges represent propagation relationships.
- Each node contains 10-dimensional profile features.

## Target

Binary graph-level classification:

- Class 0: Real
- Class 1: Fake

## Model

GraphSAGE

Architecture:

Input features
↓
GraphSAGE Layer 1
↓
ReLU
↓
GraphSAGE Layer 2
↓
Global Mean Pooling
↓
Fully Connected Layer
↓
2-class output

## Training

Loss:
Cross Entropy Loss

Optimizer:
Adam

Initial learning rate:
0.001

Batch size:
32

Initial epochs:
30

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Purpose

This experiment establishes the baseline GNN capability of the system.

The trained model will later serve as the foundation for propagation-risk analysis and the AI service.

## Expected Outcome

The model should learn graph-level patterns associated with fake and real news propagation.

The exact performance will be recorded after training.

## Notes

This is a baseline experiment and is not the final proposed intervention model.

## Experimental Results

The GraphSAGE model was trained for 30 epochs using the UPFD GossipCop dataset.

### Final Performance

| Metric                   |  Score |
| ------------------------ | -----: |
| Best Validation Accuracy | 91.39% |
| Test Accuracy            | 89.94% |
| Precision                | 90.78% |
| Recall                   | 88.94% |
| F1-score                 | 89.85% |
| ROC-AUC                  | 96.00% |

### Confusion Matrix

| Actual / Predicted | Real | Fake |
| ------------------ | ---: | ---: |
| Real               | 1737 |  173 |
| Fake               |  212 | 1704 |

### Interpretation

The GraphSAGE baseline achieved 89.94% test accuracy and an F1-score of 89.85%, indicating balanced performance across the Real and Fake classes.

The model achieved a ROC-AUC of 96.00%, demonstrating strong discrimination between the two classes.

The relatively balanced precision and recall values indicate that the model does not strongly favor either class.

The confusion matrix shows that 1,737 real-news propagation graphs and 1,704 fake-news propagation graphs were correctly classified.

### Conclusion

Experiment 01 successfully established a strong Graph Neural Network baseline for fake-news propagation graph classification. The trained model is saved as:

models/graphsage_gossipcop.pt

This model will be used as the initial GNN component of the AI service.
