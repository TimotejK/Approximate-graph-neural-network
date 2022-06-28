# Read the data
import csv
from datetime import datetime

import numpy as np
import pandas
import random
import spacy
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.graphgym import optim

from implementation.gnn import GNNStack

nlp = spacy.load("en_core_web_lg")
vector_size = 300

columns_name = ['text', 'class']
data_yelp = pandas.read_csv('../data/yelp_labelled.txt', sep='\t', header=None)
data_yelp.columns = columns_name

def text_to_graph(text, y):
    edge_index = []  # list of all edges in a graph
    doc = nlp(text)
    root = 0
    for token in doc:
        edge_index.append([token.i, token.i])  # add a self loop
        if token.i == token.head.i:
            root = token.i
        else:
            edge_index.append([token.i, token.head.i])  # add a connection from token to its parent
            edge_index.append([token.head.i, token.i])  # add a reverse connection
    x = torch.tensor(np.array([d.vector for d in doc]))  # compute token embedings
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # torch geometric expects to get the edges in a matrix with to rows and a column for each connection
    data = Data(x=x, edge_index=edge_index, y=torch.tensor([y]), text=text, root_index=root)
    return data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        # the pandas dataframe on the input contains a column labeled 'text' with yelp review texts
        # the column labeled 'class' contains a number representing the polarity of the review (positive / negative)
        self.graphs = []
        self.num_node_features = vector_size
        self.num_classes = 2
        for text, y in zip(df['text'], df['class']):
            self.graphs.append(text_to_graph(text, y))
        random.Random(1).shuffle(self.graphs)

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

# Construct the dataset.
dataset = Dataset(data_yelp)
print(len(dataset))

gnn = GNNStack(300, 10, 10)
print(gnn(dataset[0]))

def train(dataset, pooling_type, dropout=0.25, weight_decay=0, precise_training=True, w=4, csv_writer=None):
    data_size = len(dataset)
    loader = DataLoader(dataset[:int(data_size * 0.8)], batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset[int(data_size * 0.8):], batch_size=64, shuffle=True)

    # build model
    model = GNNStack(max(dataset.num_node_features, 1), 16, dataset.num_classes)
    opt = optim.Adam(model.parameters(), lr=0.0001, weight_decay=weight_decay)

    # train
    for epoch in range(101):
        total_loss = 0
        model.train()
        model.apply(lambda m: setattr(m, 'precise_computation', precise_training))
        model.apply(lambda m: setattr(m, 'parameter_w', w))
        for batch in loader:
            opt.zero_grad()
            pred = model(batch)
            label = batch.y
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)

        if epoch % 10 == 0:
            test_acc, test_prec, test_rec, test_f1 = test(test_loader, model, precise=False)
            print("Approximate: Epoch {}. Loss: {:.4f}. Test accuracy: {:.4f}".format(
                epoch, total_loss, test_acc))
            writer.writerow([epoch, w, False, precise_training, test_acc, test_prec, test_rec, test_f1])
            test_acc, test_prec, test_rec, test_f1 = test(test_loader, model, precise=True)
            print("Exact: Epoch {}. Loss: {:.4f}. Test accuracy: {:.4f}".format(
                epoch, total_loss, test_acc))
            writer.writerow([epoch, w, True, precise_training, test_acc, test_prec, test_rec, test_f1])

    return model

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
def test(loader, model, precise=True):
    model.eval()
    model.apply(lambda m: setattr(m, 'precise_computation', precise))
    predictions = []
    labels = []
    with torch.no_grad():
        for data in loader:
            pred = model(data)
            pred = pred.argmax(dim=1)
            label = data.y
            predictions += pred.tolist()
            labels += label.tolist()

    total = len(loader.dataset)
    accuracy = accuracy_score(labels, predictions)
    recall = recall_score(labels, predictions)
    precision = precision_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return accuracy, precision, recall, f1

classification_type = 'pooling'
if __name__ == '__main__':
    log = open('results' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.csv', 'w')
    writer = csv.writer(log)
    writer.writerow(['epoch', 'w', 'precise', 'precise training', 'accuracy', 'precision', 'recal', 'f1'])
    for w in range(3, 8):
        model = train(dataset, classification_type, dropout=0.25, weight_decay=0.001, precise_training=True, w=w, csv_writer=writer)
    log.close()
