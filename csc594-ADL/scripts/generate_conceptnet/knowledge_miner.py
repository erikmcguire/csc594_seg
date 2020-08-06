from copy import deepcopy
import pandas as pd
import numpy as np
import os.path

class KnowledgeMiner:

    def __init__(self, dev_data_path, device, Template, bert, **kwarg):
        """ Creates a class instance for doing KBC with a given template and
        HuggingFace bert model. Template classes defined in `sentences.py` """
        self.sentences = Template(
            dev_data_path,
            device,
            **kwarg
        )
        bert.eval()
        bert.to(device)
        self.bert = bert
        self.device = device
        self.results = []
        print("Initialized sentences.")

    def make_predictions(self):
        data = []
        n = len(self.sentences)
        for idx, (sent) in enumerate(self.sentences):
            sent_str = self.sentences.id_to_text(sent)
            data.append(sent_str)
            if idx % 100 == 0:
                print("{}%".format(idx/n * 100))
                print(sent_str)
        print("Writing sentences to file...")
        filepathname = os.path.join("../", "sentences_train.txt")
        with open(filepathname, "w") as outfile:
            outfile.write("\n".join(data))
        return data
