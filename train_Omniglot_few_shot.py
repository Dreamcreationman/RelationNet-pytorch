import os, sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

import math
import torch
import random
import argparse
import numpy as np
from torch import nn
from torch.optim import Adam
from torch.nn.functional import one_hot
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torch.utils.data import DataLoader
from preprocess import split_train_test
from torch.utils.tensorboard import SummaryWriter
from data.Omniglot import Omniglot
from data.OmniglotTask import OmniglotTask
from model.model import EmbeddingBlock, RelationBlock

# console args parser
parser = argparse.ArgumentParser(description="Omniglot Few shot")
parser.add_argument("-r", "--root_dir", type=str, default="/home/data/dataset/omniglot/processed", help="root dir to omniglot")
parser.add_argument("-e", "--train_episode", type=int, default=1000000, help="number of meta training episode")
parser.add_argument("-te", "--test_episode", type=int, default=1000, help="number of meta testing episode")
parser.add_argument("-nq", "--num_per_class_query", type=int, default=15, help="number of images in query set per episode")
parser.add_argument("-n", "--n_way", type=int, default=5, help="number of ways")
parser.add_argument("-k", "--k_shot", type=int, default=5, help="number of images in support set per episode/k shot")
parser.add_argument("-lr", "--learning_rate", type=int, default=1e-3, help="learning rate")
parser.add_argument("-nf", "--num_features", type=int, default=64, help="number of features in convolutional layer")
parser.add_argument("-fd", "--fc_dim", type=int, default=8, help="dim of fc layers")
args = parser.parse_args()

# Parameters
print("Parsing args")
ROOT_DIR = args.root_dir
TRAINING_EPISODE = args.train_episode
TESTING_EPISODE = args.test_episode
NUM_QUERY = args.num_per_class_query
N_WAY = args.n_way
K_SHOT = args.k_shot
LEARNING_RATE = args.learning_rate
NUM_FEATURES = args.num_features
FC_DIM = args.fc_dim
tensorboard_log =  "tensorboard/"
if not os.path.exists(tensorboard_log):
    os.mkdir(tensorboard_log)
writer = SummaryWriter(tensorboard_log)
meta_train_folder, meta_test_class = split_train_test(ROOT_DIR)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


# Define network
print("Defining network parameters")
embed = EmbeddingBlock(in_channels=1, num_features=NUM_FEATURES).apply(weights_init).cuda()
relation = RelationBlock(num_features=NUM_FEATURES, fc_dim=FC_DIM).apply(weights_init).cuda()
embed_para = sum(param.numel() for param in embed.parameters())
relation_para = sum(param.numel() for param in relation.parameters())
print('Number of parameters:', embed_para+relation_para)

embed_optim = Adam(embed.parameters(), lr=LEARNING_RATE)
embed_scheduler = StepLR(embed_optim, step_size=100000, gamma=0.5)
relation_optim = Adam(relation.parameters(), lr=LEARNING_RATE)
relation_scheduler = StepLR(relation_optim, step_size=100000, gamma=0.5)
criterion = nn.MSELoss().cuda()

last_accuracy = 0.0
print("Start Training")
for episode in range(TRAINING_EPISODE):

    transform = transforms.Compose([
        random.choice([
            transforms.RandomRotation([90, 90]),
            transforms.RandomRotation([180, 180]),
            transforms.RandomRotation([270, 270]),
        ]),
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.92206],
                             std=[0.08426])
    ])
    task = OmniglotTask(meta_train_folder, num_class=N_WAY, num_shot=K_SHOT, num_query=NUM_QUERY)
    support_dataset = Omniglot(task=task, support=True, transfrom=transform)
    query_dataset = Omniglot(task=task, support=False, transfrom=transform)
    support_loader = DataLoader(support_dataset, batch_size=task.num_class * task.num_shot, shuffle=False)
    query_loader = DataLoader(query_dataset, batch_size=task.num_class * task.num_query, shuffle=True)

    support_image, _ = support_loader.__iter__().next() # [num_class*num_per_class x 3 x 28 x 28]
    query_image, query_label = query_loader.__iter__().next()
    support_image = support_image.cuda()
    query_image = query_image.cuda()
    query_label = one_hot(query_label.long(), num_classes=N_WAY).float().cuda()

    support_embed_feature = embed(support_image) # [num_class*num_per_class x num_features x 8 x 8]
    support_embed_feature = support_embed_feature.view(N_WAY, K_SHOT, NUM_FEATURES, 8, 8)
    support_embed_feature = torch.sum(support_embed_feature, dim=1).squeeze(1) # [num_class x num_features x 8 x 8]
    support_embed_feature = support_embed_feature.unsqueeze(0).repeat(N_WAY*NUM_QUERY, 1, 1, 1, 1) # [num_class*num_query x num_class x num_features x 8 x 8]

    query_embed_feature = embed(query_image) # [num_class*num_per_class x num_features x 3 x 3]
    query_embed_feature = query_embed_feature.unsqueeze(0).repeat(N_WAY, 1, 1, 1, 1).transpose(0, 1) # [num_class*num_per_class x num_class x num_features x 8 x 8]
    concat_feature = torch.cat([support_embed_feature, query_embed_feature], dim=2).view(-1, 2 * NUM_FEATURES, 8, 8) # [num_class*num_per_class*num_class x 2*num_features x 8 x 8]
    out = relation(concat_feature).view(-1, N_WAY)

    loss = criterion(query_label, out)
    embed_optim.zero_grad()
    relation_optim.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm(embed.parameters(), 0.5)
    torch.nn.utils.clip_grad_norm(relation.parameters(), 0.5)

    embed_scheduler.step(episode)
    relation_scheduler.step(episode)
    embed_optim.step()
    relation_optim.step()

    if (episode + 1) % 100 == 0:
        print("Training Episode:{}/{}        loss:{}".format(episode+1, TRAINING_EPISODE, float(loss.item())))
        writer.add_scalar("Training Loss", loss.item(), episode+1)

    if (episode + 1) % 5000 == 0:
        print("Testing....")
        total_rewards = 0
        rewards_list = []
        for i in range(TESTING_EPISODE):
            task = OmniglotTask(meta_test_class, num_class=N_WAY, num_shot=K_SHOT, num_query=K_SHOT)
            support_dataset = Omniglot(task=task, support=True, transfrom=transform)
            query_dataset = Omniglot(task=task, support=False, transfrom=transform)
            support_loader = DataLoader(support_dataset, batch_size=task.num_class * task.num_shot, shuffle=False)
            query_loader = DataLoader(query_dataset, batch_size=task.num_class * task.num_query, shuffle=True)

            support_image, _ = support_loader.__iter__().next()  # [num_class*num_per_class x 3 x 28 x 28]
            query_image, query_label = query_loader.__iter__().next()
            support_image = support_image.cuda()
            query_image = query_image.cuda()

            support_embed_feature = embed(support_image)  # [num_class*num_per_class x num_features x 8 x 8]
            support_embed_feature = support_embed_feature.view(N_WAY, K_SHOT, NUM_FEATURES, 8, 8)
            support_embed_feature = torch.sum(support_embed_feature, dim=1).squeeze(
                1)  # [num_class x num_features x 8 x 8]
            support_embed_feature = support_embed_feature.unsqueeze(0).repeat(N_WAY * K_SHOT, 1, 1, 1,
                                                                              1)  # [num_class*num_query x num_class x num_features x 8 x 8]

            query_embed_feature = embed(query_image)  # [num_class*num_per_class x num_features x 3 x 3]
            query_embed_feature = query_embed_feature.unsqueeze(0).repeat(N_WAY, 1, 1, 1, 1).transpose(0,
                                                                                                       1)  # [num_class*num_per_class x num_class x num_features x 8 x 8]
            concat_feature = torch.cat([support_embed_feature, query_embed_feature], dim=2).view(-1, 2 * NUM_FEATURES,
                                                                                                 8,
                                                                                                 8)  # [num_class*num_per_class*num_class x 2*num_features x 8 x 8]
            out = relation(concat_feature).view(-1, N_WAY)
            _, predict_labels = torch.max(out.data, 1)
            rewards = [1 if predict_labels[j]==query_label[j] else 0 for j in range(N_WAY*K_SHOT)]
            rewards_list.append(rewards)
            total_rewards += np.sum(rewards)

        test_accuracy = total_rewards / 1.0 / N_WAY / K_SHOT / TESTING_EPISODE
        print("Test accuracy:", test_accuracy)
        print("reward list:", rewards_list)
        writer.add_scalar("Test/Accuracy", test_accuracy, episode)
        if test_accuracy > last_accuracy:

            # save networks
            torch.save(embed.state_dict(),str("./model/omniglot_feature_encoder_" + str(N_WAY) +"way_" + str(K_SHOT) +"shot.pkl"))
            torch.save(relation.state_dict(),str("./model/omniglot_relation_network_"+ str(N_WAY) +"way_" + str(K_SHOT) +"shot.pkl"))

            print("save networks for episode:", episode)

            last_accuracy = test_accuracy

        print("Start Training")
