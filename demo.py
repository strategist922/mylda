import os

from mylda import LDA, demo_dataset_dir

filenames = [os.path.join(demo_dataset_dir, x)
             for x in os.listdir(demo_dataset_dir)]
documents = [open(x).read() for x in filenames]

lda_model = LDA(K=5, n_early_stop=20)
lda_model.fit(documents, max_dict_len=5000, stem=False)
lda_model.show_topic(topNum=15)
