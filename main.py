import os
import shutil
from typing import List
import numpy as np
from PIL import Image
from catboost import CatBoostClassifier
from insightface import *
from insightface.app import FaceAnalysis
from sklearn.metrics import accuracy_score
from numpy import exp

trashhold = 80

def rename_files(dir):
    for subdir in os.listdir(dir):
        path = dir + subdir + '/'
        if len(os.listdir(path)) > 1:
            filenames = [filename for filename in os.listdir(path)]
            for fname in filenames:
                if subdir not in fname:
                    face_path_old = path + fname
                    face_path_new = path + subdir + '_' + fname
                    shutil.move(face_path_old, face_path_new)


def get_test_data(dir):
    for subdir in os.listdir(dir):
        path = dir + subdir + '/'
        if len(os.listdir(path)) > 1:
            filenames = [filename for filename in os.listdir(path)]
            face_path = path + filenames[0]
            shutil.move(face_path, 'dataset/test_faces/')


def extract_face(filename, required_size=(256, 256)):
    image = Image.open(filename)
    img_arr = np.array(image)
    im = Image.fromarray((img_arr))
    im = im.resize(required_size)
    rgb_arr = np.array(im.convert('RGB'))
    emb_res = app.get(rgb_arr)
    try:
        face_array = emb_res[0].embedding
        return face_array
    except:
        print('no embedding found for this image')


def load_face(dir):
    faces = list()
    for filename in os.listdir(dir):
        path = dir + filename
        face = extract_face(path)
        faces.append(face)
    return faces


def load_dataset(dir):
    X, y = list(), list()
    i = 1
    for subdir in os.listdir(dir):
        path = dir + subdir + '/'
        faces = load_face(path)
        labels = [subdir for i in range(len(faces))]
        print("loaded %d sample for class: %s" % (len(faces), subdir))
        X.extend(faces)
        y.extend(labels)
        i += 1
    return np.array(X), np.array(y)


def filter_empty_embs(img_set: np.array, img_labels: List[str]):
    good_idx = [i for i, x in enumerate(img_set) if x is not None]
    clean_labels = img_labels[good_idx]
    clean_embs = img_set[good_idx]
    return clean_embs, clean_labels


if __name__ == "__main__":
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_thresh=0.5, det_size=(256, 256))

    rename_files('dataset/full_train/')

    clf_model = CatBoostClassifier(iterations=200,
                                   task_type="GPU",
                                   devices='0:1',
                                   learning_rate=0.01,
                                   )

    print("1. Обучить модель")
    print("2. Проверить")
    userInput = input();
    if userInput == '1':
        get_test_data('dataset/full_train/')
        trainX, trainY = load_dataset('dataset/full_train/')

        assert len(trainX) == len(trainY)
        train_emb, train_labels = filter_empty_embs(trainX, trainY)

        assert len(train_emb) == len(train_labels)
        print("Train_X size is {} , train_y size is {} ".format(train_emb.shape, train_labels.shape))

        clf_model.fit(np.array(list(train_emb)),
                      train_labels,
                      verbose=False,
                      plot=True)

        clf_model.save_model("my_model")
    elif userInput == '2':
        clf_model.load_model("my_model")

    preds = []
    true_labels = []
    good = 0
    all = 0
    for filename in os.listdir('dataset/test_faces/'):
        image = Image.open('dataset/test_faces/' + filename)
        img_arr = np.array(image)
        im = Image.fromarray((img_arr))
        required_size = (256, 256)
        im = im.resize(required_size)
        rgb_arr = np.array(im.convert('RGB'))
        emb_res = app.get(rgb_arr)
        try:
            face_array = emb_res[0].embedding
        except:
            print('no embedding found for this image')

        predict = clf_model.predict(face_array)

        raw_pred = clf_model.predict(
            face_array,
            prediction_type='RawFormulaVal'
        )

        if(max(raw_pred)==0.):
            raw_pred = raw_pred - min(raw_pred)

        sigmoid = lambda x: 1 / (1 + exp(-x))
        probabilities = sigmoid(raw_pred)

        if max(probabilities) > trashhold/100:
            print(predict + "(" + str(max(probabilities) * 100) + "%) - " + filename)
            if predict[0] in filename:
                good += 1
        else:
            print("unknown - " + filename)
            if predict[0] not in filename:
                good += 1
        all += 1

    print("точность " + str(good * 100 / all ))