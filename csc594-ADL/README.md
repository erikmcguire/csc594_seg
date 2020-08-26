> We must navigate to the main project folder in mounted My Drive. 

> Assumes the following structure:
<pre>.
├── content
│   ├──drive                         # Mounted drive folder.
│   │   └── My Drive                 # Mounted drive folder.
│   │       └── CSC-594-ADL          # Main project folder.
│   │           ├── datasets         # ConceptNet and ROCStories.
│   │           ├── endings          # Correct and generated endings per model.
│   │           ├── evals            # Evaluation results for stories and endings per model.
│   │           ├── models           # Pretrained models, tokenizers, vocabulary, etc.
│   │           ├── scripts          # Scripts for training and generation.
│   │           └── stories          # Combined story bodies and generated endings per model.
│   ├── sample_data                  # Default Colab folder.
│   └── transformers                 # Installed from HuggingFace.
└── ...
</pre>
