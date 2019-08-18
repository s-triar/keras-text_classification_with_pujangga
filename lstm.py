from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Embedding, SpatialDropout1D, LSTM, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.utils import to_categorical
import matplotlib.pyplot as plt

def wordEmbedding(vocab_size, max_len_pad, data, kolom):
    encoded_docs = [one_hot(d, vocab_size) for d in data[kolom]]
    padded_docs = pad_sequences(encoded_docs, maxlen=max_len_pad, padding='post')
    return padded_docs

def defineModel(vocab_size, embed_size, dropout_ratio, n_hidden_units, shape, n_kelas, n_lstm_layer ):
    model = Sequential()
    model.add(Embedding(vocab_size, embed_size, input_length=shape[1]))
    model.add(LSTM(units=n_hidden_units, return_sequences=True))
    for i in range(n_lstm_layer-2):
        if(i % 2 == 0):
            model.add(LSTM(units=n_hidden_units,
                           return_sequences=True, dropout=dropout_ratio))
        else:
            model.add(LSTM(units=n_hidden_units, return_sequences=True))
    model.add(LSTM(units=n_hidden_units))
    model.add(Dense(n_hidden_units+1024, activation='relu'))
    model.add(Dropout(dropout_ratio))
    model.add(Dense(n_kelas, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def runModel(model, data_x, data_y, n_epoch, batch_size, name, data_validatation_x, data_validation_y):
    tfboard = TensorBoard(log_dir="logs/{}".format(name), histogram_freq=0,
                          batch_size=batch_size, write_graph=True)
    filepath = name+".h5"
    checkPoint = ModelCheckpoint(
        "./models/checkpoints/"+filepath, verbose=1, save_best_only=True, mode="max", monitor="val_acc")
    model.fit(data_x, data_y, epochs=n_epoch, batch_size=batch_size, validation_data=(data_validatation_x, data_validation_y),
              callbacks=[tfboard, checkPoint])
    model.save("./models/"+name+".h5")
    model.summary()
    scores = model.evaluate(data_validatation_x, data_validation_y, verbose=0)
    print("score training: ", scores)
    return model


def importModel(path):
    return load_model(path)


def predictVal(model, dataX, dataY):
    prediction = model.predict(dataX)
    scores = model.evaluate(dataX, dataY, verbose=0)
    print("score prediction: ", scores)
    return prediction
