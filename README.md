# Commonsense Story Ending Generation
Project for CSC 594: Topics in AI: Advanced Deep Learning (Winter 2020) at DePaul University.

* [Course presentation video](https://drive.google.com/file/d/1bjyqaGZO7YDmgxQyvQ74JLSKCqcMA7pI/view?usp=sharing)
* [Write-up](https://github.com/erikmcguire/common_seg/blob/master/csc594-ADL/documents/csc594-810-mcguire_erik-project-report.pdf)

## Abstract

Story Ending Generation (SEG) is a Natural Language Generation task that seeks to generate coherent conclusions to brief stories. GPT-2 is a large pretrained causal language model which requires minimal task-specific changes to the architecture for supervised fine-tuning. As it is a relatively new generative model that has outperformed previous state-of-the-art results on a wide range of commonsense reasoning tasks, this project uses GPT-2 to attack the SEG problem by incorporation of an external knowledge graph as well as multi-task fine-tuning on the ROCStories dataset, a corpus of brief stories designed to capture causal and temporal relationships. External knowledge is injected through language modeling on sentences generated from ConceptNet relations. The multi-task fine-tuning setup uses discriminative tasks, namely the Story Cloze Task (SCT) and/or Sentiment Matching (SM), complemented by an auxiliary language modeling objective. A variety of automated evaluation methods are used to compare the endings generated by permutations of the base model engendered by these training regimes. Preliminary results show that the incorporation of ConceptNet via a dataset of 600k samples improves fine-tuning losses, but less so than further fine-tuning with the ROCStories dataset. Evaluations of generation results show little to no benefit for training beyond fine-tuning the base model for the Story Cloze Task.
