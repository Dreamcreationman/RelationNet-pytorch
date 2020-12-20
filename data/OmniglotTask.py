import os, sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

import random
import numpy as np


class OmniglotTask():
    def __init__(self, class_folder, num_class=5, num_shot=5, num_query=15):
        self.num_class = num_class
        self.num_shot = num_shot
        self.num_query = num_query
        self.class_folder = class_folder
        self.tasks = []
        self.tasks = random.sample(self.class_folder, self.num_class)
        labels = ["-".join((t.split("/")[-2], t.split("/")[-1])) for t in self.tasks]
        self.labels = dict(zip(labels, np.array(range(len(self.tasks)))))
        self.support_root = []
        self.query_root = []
        for t in self.tasks:
            temp = [os.path.join(t, dir) for dir in os.listdir(t)]
            random.seed(1)
            random.shuffle(temp)
            self.support_root += temp[:self.num_shot]
            self.query_root += temp[self.num_shot: self.num_shot+self.num_query]
        self.support_labels = [self.labels["-".join((s.split("/")[-3], s.split("/")[-2]))] for s in self.support_root]
        self.query_labels = [self.labels["-".join((s.split("/")[-3], s.split("/")[-2]))] for s in self.query_root]