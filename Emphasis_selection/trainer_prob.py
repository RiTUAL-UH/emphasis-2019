from sklearn.metrics import f1_score
import time
import torch
import torch.nn.functional as F
import numpy as np
import random
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
# from sklearn.metrics import f1_score
from itertools import chain
from sklearn.metrics import roc_auc_score
from helper import Helper
import config
from logger import Logger
import itertools
from visualization import attention_visualization
from sklearn_crfsuite import metrics
import pickle
import os
helper = Helper()
logger = Logger(config.output_dir_path + 'logs')

def tensor_logging(model, info, epoch):
    for tag, value in info.items():
        logger.log_scalar(tag, value, epoch + 1)
    # Log values and gradients of the model parameters
    for tag, value in model.named_parameters():
        if value.grad is not None:
            tag = tag.replace('.', '/')
            if torch.cuda.is_available():
                logger.log_histogram(tag, value.data.cpu().numpy(), epoch + 1)
                logger.log_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)

def check_predictions(preds, targets, mask):
    overlaped = (preds == targets)
    right = np.sum(overlaped * mask)
    total = mask.sum()
    return right, total, (overlaped * mask)

def visualize_attention(wts,words,filename):
    """
    Visualization function to create heat maps for prediction, ground truth and attention (if any) probabilities
    :param wts:
    :param words:
    :param filename:
    :return:
    """
    wts_add = wts.cpu()
    wts_add_np = wts_add.data.numpy()
    wts_add_list = wts_add_np.tolist()
    text= []
    for index, test in enumerate(words):
        text.append(" ".join(test))
    attention_visualization.createHTML(text, wts_add_list, filename)
    return

def get_batch_all_label_pred(numpy_predictions, numpy_label, mask_numpy, scores_numpy=None):
    """
    To remove paddings
    :param numpy_predictions:
    :param numpy_label:
    :param mask_numpy:
    :param scores_numpy: need this for computing ROC curve
    :return:
    """
    all_label =[]
    all_pred =[]
    all_score = []
    for i in range(len(mask_numpy)):

        all_label.append(list(numpy_label[i][:mask_numpy[i].sum()]))
        all_pred.append(list(numpy_predictions[i][:mask_numpy[i].sum()]))
        if isinstance(scores_numpy, np.ndarray):
            all_score.append(list(scores_numpy[i][:mask_numpy[i].sum()]))

        assert(len(list(numpy_label[i][:mask_numpy[i].sum()]))==len(list(numpy_predictions[i][:mask_numpy[i].sum()])))
        if isinstance(scores_numpy, np.ndarray):
            assert(len(list(numpy_label[i][:mask_numpy[i].sum()])) == len(list(scores_numpy[i][:mask_numpy[i].sum()])))
        assert(len(all_label)==len(all_pred))
    return  (all_label, all_pred) if not isinstance(scores_numpy, np.ndarray) else (all_label, all_pred, all_score)

def to_tensor_labels(encodings,  return_mask=False):
    maxlen = max(map(len, encodings))
    tensor =[]
    for i, sample in enumerate(encodings):
        seq_len = len(sample)
        padding_len = abs(seq_len - maxlen)
        pad = [[1,0]] * padding_len
        sample.extend(pad)

        tensor.append(sample)
    tensor_tens = torch.Tensor(tensor)

    if torch.cuda.is_available():
        tensor_tens = tensor_tens.cuda()
    return  tensor_tens

def to_tensor(encodings, pad_value=0, return_mask=False):
    maxlen = max(map(len, encodings))
    tensor = torch.zeros(len(encodings), maxlen).long() + pad_value
    mask = torch.zeros(len(encodings), maxlen).long()
    for i, sample in enumerate(encodings):
        tensor[i, :len(sample)] = torch.tensor(sample, dtype=torch.long)
        mask[i, :len(sample)] = 1
    if torch.cuda.is_available():
        tensor = tensor.cuda()
        mask = mask.cuda()
    return (tensor, mask) if return_mask else tensor



class Trainer(object):
    def __init__(self, corpus, encoder, batch_size, epochs):
        self.corpus = corpus
        self.encoder = encoder
        self.batch_size = batch_size
        self.batch_size_org = batch_size
        self.epochs = epochs
        self.PAD_TARGET_IX = 0

    def batchify(self, batch_i, dataset, model):
        """
        :param batch_i: ith batch
        :param dataset: train, dev or test set
        :param model: model
        :param theLoss: loss
        :return:
        """

        batch_start = batch_i * self.batch_size_org
        batch_end = batch_start + self.batch_size
        l_tensor = to_tensor_labels(dataset.labels[batch_start: batch_end])
        words = dataset.words[batch_start: batch_end]

        if config.if_Elmo:
            scores, mask, att_w = model.forward(words)
            actual_words_no_pad = words
        else:
            w_tensor, mask= to_tensor(words,  return_mask=True )
            scores, att_w = model.forward(w_tensor, mask) # scores before flatten shape:  [batch_size, seq_len, num_labels]
            w_no_pad = w_tensor.cpu().detach().numpy()
            actual_words_no_pad = [[self.encoder.index2word[elem] for elem in elems] for elems in w_no_pad]

        batch_size, seq_len = l_tensor.size(0), l_tensor.size(1) # target_shape: [batch_size, seq_len]

        scores_flat = F.log_softmax(scores.view(batch_size * seq_len, -1), dim=1) # score_flat shape = [batch_size * seq_len, num_labels]

        target_flat = l_tensor.view(batch_size * seq_len, 2)  # target_flat shape= [batch_size * seq_len]

        return scores, l_tensor, scores_flat, target_flat, seq_len, mask, words,actual_words_no_pad, att_w

    def train(self, model, theLoss, optimizer):
        """
        The train function
        :param model:
        :param theLoss:
        :param optimizer:
        :return:
        """

        print("==========================================================")
        print("[LOG] Training model...")

        total_batch_train = len(self.corpus.train.labels) // self.batch_size
        total_batch_dev = len(self.corpus.dev.labels) // self.batch_size

        if (len(self.corpus.train.labels)) % self.batch_size > 0:
            total_batch_train += 1

        if len(self.corpus.dev.labels) % self.batch_size > 0:
            total_batch_dev += 1


        for epoch in range(self.epochs):

            self.batch_size = self.batch_size_org
            train_total_preds = 0
            train_right_preds = 0
            total_train_loss =0
            model.train()
            train_total_y_true = []
            train_total_y_pred =[]
            with open("output_train.txt", "w") as f:
                for batch_i in range(total_batch_train):

                    if (batch_i == total_batch_train - 1) and (len(self.corpus.train.labels) % self.batch_size > 0):
                        self.batch_size = len(self.corpus.train.labels) % self.batch_size
                    optimizer.zero_grad()
                    score, target, scores_flat, target_flat, seq_len, mask, words,__, _= self.batchify(batch_i, self.corpus.train, model)
                    train_loss = theLoss(scores_flat, F.softmax(target_flat,dim=1))#/ self.batch_size
                    target_flat_softmaxed = F.softmax(target_flat, 1)



                    train_loss.backward()
                    optimizer.step()
                    total_train_loss += train_loss.item() * self.batch_size


                    _, predictions_max = torch.max(torch.exp(scores_flat), 1)
                    predictions_max = predictions_max.view(self.batch_size, seq_len)
                    numpy_predictions_max = predictions_max.cpu().detach().numpy()


                    _, label_max = torch.max(target_flat_softmaxed, 1)
                    label_max = label_max.view(self.batch_size, seq_len)
                    numpy_label_max = label_max.cpu().detach().numpy()


                    #mask:
                    mask_numpy = mask.cpu().detach().numpy()
                    right, whole, overlaped = check_predictions(numpy_predictions_max, numpy_label_max, mask_numpy)
                    train_total_preds += whole
                    train_right_preds += right
                    all_label, all_pred = get_batch_all_label_pred(numpy_predictions_max, numpy_label_max, mask_numpy)
                    train_total_y_pred.extend(all_pred)
                    train_total_y_true.extend(all_label)

                   


            train_f1_total = metrics.flat_f1_score(train_total_y_true, train_total_y_pred, average= "micro")

            train_loss = total_train_loss/ len(self.corpus.train.labels)
            print("[lOG] ++Train_loss: {}++, ++MAX train_accuracy: {}++, ++MAX train_f1_score: {}++ ".format(train_loss, (train_right_preds / train_total_preds), (train_f1_total) ))




            print("[LOG] ______compute dev: ")
            model.eval()
            self.batch_size = self.batch_size_org
           
            dev_right_preds = 0
            dev_total_preds = 0
            total_dev_loss = 0
            dev_total_y_true = []
            dev_total_y_pred = []
            for batch_i in range(total_batch_dev):
                if (batch_i == total_batch_dev - 1) and (len(self.corpus.dev.labels) % self.batch_size > 0):
                    self.batch_size = len(self.corpus.dev.labels) % self.batch_size
                    
                dev_score, dev_target,dev_scores_flat, dev_target_flat, dev_seq_len, dev_mask, dev_words,__, _= self.batchify(batch_i, self.corpus.dev, model)
                dev_loss = theLoss(dev_scores_flat, F.softmax(dev_target_flat, 1)) #/ self.batch_size
                total_dev_loss += dev_loss.item() * self.batch_size
                dev_target_flat_softmaxed = F.softmax(dev_target_flat, 1)

                _, dev_predictions_max = torch.max(dev_scores_flat, 1)
                dev_predictions_max = dev_predictions_max.view(self.batch_size, dev_seq_len)
                dev_numpy_predictions_max = dev_predictions_max.cpu().detach().numpy()


                _, dev_label_max = torch.max(dev_target_flat_softmaxed, 1)
                dev_label_max = dev_label_max.view(self.batch_size, dev_seq_len)
                dev_numpy_label_max = dev_label_max.cpu().detach().numpy()


                # mask:
                dev_mask_numpy = dev_mask.cpu().detach().numpy()

                dev_right, dev_whole, dev_overlaped = check_predictions(dev_numpy_predictions_max, dev_numpy_label_max, dev_mask_numpy)
                dev_total_preds += dev_whole
                dev_right_preds += dev_right

                all_label, all_pred = get_batch_all_label_pred(dev_numpy_predictions_max, dev_numpy_label_max, dev_mask_numpy, 0)
                dev_total_y_pred.extend(all_pred)
                dev_total_y_true.extend(all_label)


            else:
                dev_f1_total_micro = metrics.flat_f1_score(dev_total_y_true, dev_total_y_pred, average= "micro")
            dev_loss = total_dev_loss / len(self.corpus.dev.labels)
            dev_f1_total_macro = metrics.flat_f1_score(dev_total_y_true, dev_total_y_pred, average="macro")

            #checkpoint:
            is_best = helper.checkpoint_model(model, optimizer, config.output_dir_path, dev_loss, epoch + 1, 'min')

            print("<<dev_loss: {}>> <<dev_accuracy: {}>> <<dev_f1: {}>> ".format( dev_loss, (dev_right_preds / dev_total_preds), (dev_f1_total_micro)))
            print("--------------------------------------------------------------------------------------------------------------------------------------------------")
            #tensorBoard:
            info = {'training_loss': train_loss,
                    'train_accuracy': (train_right_preds / train_total_preds),
                    'train_f1': (train_f1_total),
                    'validation_loss': dev_loss,
                    'validation_accuracy': (dev_right_preds / dev_total_preds),
                    'validation_f1_micro': (dev_f1_total_micro),
                    'validation_f1_macro': (dev_f1_total_macro)
                    }
            tensor_logging(model, info, epoch)

    def predict(self, model, theLoss, theCorpus):
        print("==========================================================")
        print("Predicting...")
        helper.load_saved_model(model, config.output_dir_path + 'best.pth')
        model.eval()
        self.batch_size = self.batch_size_org
        total_batch_test = len(theCorpus.labels) // self.batch_size
        if len(theCorpus.words) % self.batch_size > 0:
            total_batch_test += 1

        test_right_preds, test_total_preds = 0, 0
        test_total_y_true = []
        test_total_y_pred = []
        test_total_y_scores = []
        total_scores_numpy_probs =[]
        total_labels_numpy_probs =[]
        total_mask_numpy =[]
        total_test_loss = 0
        with open("output_test.txt", "w") as f:
            for batch_i in range(total_batch_test):
                if (batch_i == total_batch_test - 1) and (len(theCorpus.words) % self.batch_size > 0):
                    self.batch_size = len(theCorpus.words) % self.batch_size
                score, target, scores_flat, target_flat, seq_len, mask, words,actual_words_no_pad,  att_w = self.batchify(batch_i, theCorpus, model)
                test_loss =  theLoss(scores_flat, F.softmax(target_flat, 1)) #/ self.batch_size

                total_test_loss += test_loss.item() * self.batch_size
                scores_flat_exp = torch.exp(scores_flat)

                print("--[LOG]-- test loss: ", test_loss)

                _, predictions_max = torch.max(scores_flat_exp, 1)

                predictions_max = predictions_max.view(self.batch_size, seq_len)
                numpy_predictions_max = predictions_max.cpu().detach().numpy()

                # computing scores for ROC curve:
                scores_numpy = scores_flat_exp[:, 1].view(self.batch_size, seq_len)
                scores_numpy = scores_numpy.cpu().detach().numpy()

                total_scores_numpy_probs.extend(scores_numpy)

                # if based on MAX
                _, label_max = torch.max(target_flat, 1)
                label_max = label_max.view(self.batch_size, seq_len)
                numpy_label_max = label_max.cpu().detach().numpy()
                # for computing senetnce leveL:
                total_labels_numpy_probs.extend(target_flat[:, 1].view(self.batch_size, seq_len).cpu().detach().numpy())

                # mask:
                mask_numpy = mask.cpu().detach().numpy()
                total_mask_numpy.extend(mask_numpy)
                right, whole, overlaped = check_predictions(numpy_predictions_max, numpy_label_max, mask_numpy)
                test_total_preds += whole
                test_right_preds += right
                all_label, all_pred, all_scores= get_batch_all_label_pred(numpy_predictions_max, numpy_label_max, mask_numpy, scores_numpy)
                test_total_y_pred.extend(all_pred)
                test_total_y_true.extend(all_label)

                #ROC:
                if config.if_ROC:
                    test_total_y_scores.extend(all_scores)

                # Visualization:
                if config.if_visualize:
                    sfe = scores_flat_exp[:, 1].view(self.batch_size, seq_len)
                    visualize_attention(sfe, actual_words_no_pad, filename='res/scores'+str(batch_i)+'.html')
                    visualize_attention(target[:,:,1], actual_words_no_pad, filename='res/target' + str(batch_i) + '.html')
                    visualize_attention(F.softmax(target, 1)[:,:,1], actual_words_no_pad, filename='res/target_softmaxed' + str(batch_i) + '.html')


            test_f1_total_micro = metrics.flat_f1_score(test_total_y_true, test_total_y_pred, average= "micro")
            test_f1_total_macro = metrics.flat_f1_score(test_total_y_true, test_total_y_pred, average="macro")
            test_f1_total_binary = metrics.flat_f1_score(test_total_y_true, test_total_y_pred, average="binary")

            roc_score= roc_auc_score(list(itertools.chain(*test_total_y_true)) , list(itertools.chain(*test_total_y_scores)))
            test_loss = total_test_loss / len(self.corpus.test.labels)

            print(
                "->>>>>>>>>>>>>TOTAL>>>>>>>>>>>>>>>>>>>>>>> test_loss: {}, test_accuracy: {}, test_f1_score_micro: {} ROC:{}".format(
                    test_loss, (test_right_preds / test_total_preds), (test_f1_total_micro), roc_score))
            print()
            print(metrics.flat_classification_report(test_total_y_true, test_total_y_pred))
            print("test_f1_total_binary: ", test_f1_total_binary)
            print("precision binary: ", metrics.flat_precision_score(test_total_y_true, test_total_y_pred, average="binary"))
            print("recall binary: ", metrics.flat_recall_score(test_total_y_true, test_total_y_pred, average="binary"))


            if not os.path.exists(config.dump_address):
                os.makedirs(config.dump_address)
            print("[LOG] dumping results in ", config.dump_address)
            pickle.dump(np.array(total_scores_numpy_probs),
                        open(os.path.join(config.dump_address, "score_pobs.pkl"), "wb"))
            pickle.dump(np.array(total_labels_numpy_probs),
                        open(os.path.join(config.dump_address, "label_pobs.pkl"), "wb"))
            pickle.dump(np.array(total_mask_numpy), open(os.path.join(config.dump_address, "mask_pobs.pkl"), "wb"))



            





