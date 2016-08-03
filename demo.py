import os

from mylda import LDA, demo_dataset_dir

filenames = [os.path.join(demo_dataset_dir, x)
             for x in os.listdir(demo_dataset_dir)]
documents = [open(x, encoding="utf8").read() for x in filenames]

# Generate corpus-based dictionary
lda_model = LDA(K=5, n_early_stop=20)
lda_model.fit(documents, max_dict_len=5000, min_tf=5, stem=False)
lda_model.show_topic(topNum=15)

# Or use standalone dictionary
lda_model = LDA(K=5, n_early_stop=20, use_default_dict=True)
# You can also use your own dictionary
# lda_model = LDA(K=5, dict_file="yourdictionary.txt")
lda_model.fit(documents)
lda_model.show_topic(topNum=15)
