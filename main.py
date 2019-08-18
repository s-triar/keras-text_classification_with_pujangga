import numpy as np
import pandas as pd
from os import path
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import PIL
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import re
import requests
import json
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import argmax
from sklearn.utils import shuffle
from math import floor
import lstm

def removeStopwords(doc, kolom):
    doc[kolom] = doc.apply(lambda x: requests.post(
        'http://localhost:9000/stopwords', json={"string": x[kolom]}).json()['data'], axis=1)
    return doc


def formalize(doc, kolom):
    doc[kolom] = doc.apply(lambda x: requests.post(
        'http://localhost:9000/formalizer', json={"string": x[kolom]}).json()['data'], axis=1)
    return doc


def openStopFile(path):
    with open(path, 'r') as f:
        return f.read()


def importData(path):
    return pd.read_json(path)


def cleaningDate(dataFrame, kol):
    new_kolom = dataFrame[kol].str.split(" ", expand=True)
    print(new_kolom)
    dataFrame["tgl"] = new_kolom[0] + " " + new_kolom[1] + \
        " " + new_kolom[2]  # +" "+new_kolom[5]
    dataFrame["jam"] = new_kolom[3] + " " + new_kolom[4]
    dataFrame.drop(columns=[kol], inplace=True)
    return dataFrame


def drawPlot(title, df, xlabel, ylabel):
    plt.figure(figsize=(15, 10))
    df.size().sort_values(ascending=False).plot.bar()
    plt.xticks(rotation=50)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def drawWordCloud(text):
    wordcloud = WordCloud(max_words=100).generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def createVocab(doc):
    fdist = FreqDist()
    for i in doc:
        for j in sent_tokenize(i):
            for word in word_tokenize(j):
                fdist[word.lower()] += 1
    return fdist


def OneHoteDecode(label_encoder, doc):
    inverted = label_encoder.inverse_transform([argmax(doc)])
    return inverted


def oneHotEncode(doc):
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(doc)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return label_encoder, onehot_encoded


def removeHashTag(doc, kolom):
    doc[kolom] = doc.apply(lambda x: re.sub(
        r'(\#[a-zA-Z0-9]*)', "", x[kolom]), axis=1)
    return doc


def removeNumberAndSymbol(doc, kolom):
    doc[kolom] = doc.apply(lambda x: re.sub(
        r'([\[\]\.\…\?\!\,0-9\/\:\"\(\)]*)', "", x[kolom]), axis=1)
    return doc


def removeUnicode(doc, kolom):
    doc[kolom] = doc.apply(lambda x: x[kolom].encode(
        'ascii', errors='ignore').strip().decode('ascii'), axis=1)
    return doc


def removeMention(doc, kolom):
    doc[kolom] = doc.apply(lambda x: re.sub(
        r'((RT\s*)*\@[a-zA-Z0-9\_]*)\s*\:*', "", x[kolom]), axis=1)
    return doc


def removeLink(doc, kolom):
    doc[kolom] = doc.apply(lambda x: re.sub(
        r'(http[a-zA-Z0-9\\\-\:\/\.]*)', "", x[kolom]), axis=1)
    return doc


def bundlingTweet(doc, kolom):
    paragraph = ""
    for d in doc[kolom]:
        paragraph += d+"."
    return paragraph


if __name__ == "__main__":

    training = False
    run_name = "project_3"
    kolom = "isi"
    kolom_label = "sentimen"
    n_kelas = 3

    max_len_pad = 10
    embed_size = 128
    n_hidden_units = 512 
    n_lstm_layers = 3
    dropout_ratio = 0.2

    n_epoch = 100
    batch_size = 32

    ratio_train = 0.7
    ratio_validation = 0.2
    ratio_test = 1 - (ratio_train + ratio_validation)

    df = importData("data/Sentiment/data_latih.json")
    df = cleaningDate(df, "tanggal")

    n_data = df.shape[0]
    n_train = floor(n_data*ratio_train)
    n_validation = floor(n_data*ratio_validation)
    n_test = n_data - (n_train+n_validation)

    print(df.head())
    print("There are {} observations and {} features in this dataset. \n".format(
        df.shape[0], df.shape[1]))
    print("There are {} users twitter in this dataset such as {} \n".format(len(df.id_user.unique()),
                                                                            [i for i in df.id_user.unique()[0:5]]))
    print("There are {} clasess this dataset \n".format(
        len(df.sentimen.unique())))

    user = df.groupby("id_user")
    print(user.describe().head())
    sentimen = df.groupby("sentimen")
    print(sentimen.describe().head())
    days = df.groupby("tgl")
    print(days.describe().head())

    drawPlot("Tweet Pengguna", user, "user id", "n tweet")
    drawPlot("Sentimen", sentimen, "kelas", "n tweet")
    drawPlot("Hari", days, "tanggal", "n tweet")

    df_clean = removeUnicode(df, kolom)
    df_clean = removeMention(df_clean, kolom)
    df_clean = removeLink(df_clean, kolom)
    df_clean = removeNumberAndSymbol(df_clean, kolom)

    # print(paragraph)
    formalized = formalize(df_clean, kolom)
    removed = removeStopwords(formalized, kolom)

    paragraph = bundlingTweet(df_clean, kolom)
    stopword_list = openStopFile("./data/Sentiment/stopword_list_TALA.txt")

    wordcloud = WordCloud(stopwords=stopword_list,
                          background_color="white").generate(paragraph)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    data_clean = shuffle(removed)

    vocab = createVocab(data_clean[kolom])
    vocab_size = len(vocab)

    LABEL_ENCODER_TRAIN, label_one_hot = oneHotEncode(
        data_clean[kolom_label])

    data_train = data_clean[:n_train]
    data_validation = data_clean[n_train:(n_train+n_validation)]
    data_test = data_clean[-n_test:]

    label_train = label_one_hot[:n_train]
    label_validation = label_one_hot[n_train:(n_train+n_validation)]
    label_test = label_one_hot[-n_test:]

    # padding word train
    word_padded_train = lstm.wordEmbedding(
        vocab_size, max_len_pad, data_train, kolom)
    # padding word validation
    word_padded_validation = lstm.wordEmbedding(
        vocab_size, max_len_pad,  data_validation, kolom)
    # padding word test
    word_padded_test = lstm.wordEmbedding(
        vocab_size, max_len_pad, data_test, kolom)

    if(training):
        model = lstm.defineModel(
            vocab_size,
            embed_size,
            dropout_ratio,
            n_hidden_units,
            word_padded_train.shape,
            n_kelas,
            n_lstm_layers
        )

        model = lstm.runModel(
            model,
            word_padded_train,
            label_train,
            n_epoch,
            batch_size,
            run_name,
            word_padded_validation,
            label_validation
        )
    else:
        model = lstm.importModel("./models/project_3.h5")

    prediction = lstm.predictVal(
        model, word_padded_test, label_test)

    # val_str = prep.OneHoteDecode(LABEL_ENCODER_TEST, label_test_one_hot[:50])
    # pred_str = prep.OneHoteDecode(LABEL_ENCODER_TEST, prediction)
    val_str = argmax(label_test, axis=1).reshape(
        (len(prediction), 1))
    pred_str = argmax(prediction, axis=1).reshape((len(prediction), 1))
    with open("hasil.txt", "w") as f:
        for real, pred, text, label in zip(val_str, pred_str, data_test[kolom], data_test[kolom_label]):
            hasil = str(pred) + "\t=>\t" + \
                str(real) + " ("+str(label)+")" + ": " + str(text)+"\n"
            f.write(hasil)