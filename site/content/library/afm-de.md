AFM-D Encoder is AriaCompute's on-device System 1 decision model: a ModernBERT-large DecisionModel
(Laya layout) that scores a state against your options and returns calibrated Choice, Score and
Noul answers. Long states keep the tail; high-cardinality Choice uses an embedding shortlist.

Pull with `ollaya pull afm-de`. Weights come from Hugging Face (`ariacompute/afm-de`) or
ModelScope (`AriaCompute/afm-de`); set `OLLAYA_HUB=modelscope` (or use a Chinese locale with
`OLLAYA_HUB=auto`) to prefer ModelScope.
