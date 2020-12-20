from PIL import Image
from torch.utils.data import Dataset


class Omniglot(Dataset):

    def __init__(self, task, support=True, transfrom=None, target_transform=None):
        super(Omniglot, self).__init__()
        self.task = task
        self.transform = transfrom
        self.target_transform = target_transform
        self.images = self.task.support_root if support else self.task.query_root
        self.labels = self.task.support_labels if support else self.task.query_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label
