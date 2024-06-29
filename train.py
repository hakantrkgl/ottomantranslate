import numpy as np  # NumPy kütüphanesi
import cv2 as cv  # OpenCV kütüphanesi
import os  # İşletim sistemi fonksiyonları için kullanılır
import re  # Düzenli ifadeler için kullanılır
import random  # Rastgele sayılar üretmek için kullanılır
# Kendi tanımladığınız utilities modülünden fonksiyonları içe aktarıyoruz.
from utilities import projection
from glob import glob  # Dosya aramak için kullanılır
from tqdm import tqdm  # İlerleme çubuğunu oluşturmak için kullanılır

from sklearn.utils import shuffle  # Veriyi karıştırmak için kullanılır
# Veri kümesini eğitim ve test alt kümelerine bölmek için kullanılır
from sklearn.model_selection import train_test_split
from sklearn import svm  # Destek Vektör Makineleri (SVM) için kullanılır
# Yapay Sinir Ağları için kullanılır
from sklearn.neural_network import MLPClassifier
# Naive Bayes sınıflandırıcı için kullanılır
from sklearn.naive_bayes import GaussianNB
# Doğruluk skoru hesaplamak için kullanılır
from sklearn.metrics import accuracy_score
import pickle  # Nesneleri serileştirmek ve deserileştirmek için kullanılır

# Sınıflandırılacak karakterler
chars = ['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف',
         'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي', 'لا', 'پ', 'چ', 'ژ', '،', 'ی']  # İşlenen karakterlerin listesi

train_ratio = 0.8  # Eğitim verisi oranı
script_path = os.getcwd()  # Kodun bulunduğu dizin
classifiers = [svm.LinearSVC(), MLPClassifier(alpha=1e-4, hidden_layer_sizes=(100,), max_iter=1000),
               MLPClassifier(alpha=1e-5, hidden_layer_sizes=(200, 100,), max_iter=1000), GaussianNB()]  # Sınıflandırıcı modelleri


# Sınıflandırıcı model isimleri
names = ['LinearSVM', '1L_NN', '2L_NN', 'Gaussian_Naive_Bayes']
selected_models = []  # Seçilen model isimlerini tutacak liste

def set_selected_models(models):
    global selected_models
    selected_models = models

width = 25  # Görüntü genişliği
height = 25  # Görüntü yüksekliği
dim = (width, height)  # Yeni boyut


def bound_box(img_char):
    # İz düşüm tabanlı sınırlayıcı kutu hesaplama
    HP = projection(img_char, 'horizontal')  # Yatay iz düşüm
    VP = projection(img_char, 'vertical')  # Dikey iz düşüm

    top = -1  # Üst sınır
    down = -1  # Alt sınır
    left = -1  # Sol sınır
    right = -1  # Sağ sınır

    i = 0
    while i < len(HP):
        if HP[i] != 0:
            top = i
            break
        i += 1

    i = len(HP)-1
    while i >= 0:
        if HP[i] != 0:
            down = i
            break
        i -= 1

    i = 0
    while i < len(VP):
        if VP[i] != 0:
            left = i
            break
        i += 1

    i = len(VP)-1
    while i >= 0:
        if VP[i] != 0:
            right = i
            break
        i -= 1

    return img_char[top:down+1, left:right+1]  # Sınırlayıcı kutuyu döndürme


def binarize(char_img):
    # Görüntüyü ikili hale getirme
    _, binary_img = cv.threshold(char_img, 127, 255, cv.THRESH_BINARY)
    binary_char = binary_img // 255  # 0 ve 1 değerlerine dönüştürme

    return binary_char  # İkili hale getirilmiş karakter görüntüsünü döndürme


def prepare_char(char_img):
    # Karakter görüntüsünü hazırlama
    binary_char = binarize(char_img)  # İkili hale getirme
    char_box = bound_box(binary_char)  # Sınırlayıcı kutu hesaplama
    # Yeniden boyutlandırma
    resized = cv.resize(char_box, dim, interpolation=cv.INTER_AREA)

    return resized  # Hazırlanmış karakter görüntüsünü döndürme


def featurizer(char_img):
    # Özellik çıkarımı
    flat_char = char_img.flatten()  # Görüntüyü düzleştirme

    return flat_char  # Düzleştirilmiş görüntüyü döndürme


def read_data(limit=4000):
    # Veri okuma işlemi
    X = []  # Girdi verisi
    Y = []  # Etiket verisi
    print("For each char")
    for char in tqdm(chars, total=len(chars)):

        folder = f'../Dataset/chars/{char}'
        char_paths = glob(f'../Dataset/chars/{char}/*.png')

        if os.path.exists(folder):
            os.chdir(folder)

            print(f'\nReading images for char {char}')
            for char_path in tqdm(char_paths[:limit], total=len(char_paths)):
                num = re.findall(r'\d+', char_path)[0]
                char_img = cv.imread(f'{num}.png', 0)
                ready_char = prepare_char(char_img)
                feature_vector = featurizer(ready_char)
                X.append(feature_vector)  # Girdi verisine ekleme
                Y.append(char)  # Etiket verisine ekleme

            os.chdir(script_path)

    return X, Y  # Girdi ve etiket verilerini döndürme


def train():
    global scores  # Skorları tutacak değişkeni global olarak tanımla
    skip = [0, 0, 0, 0]  # Tüm modelleri varsayılan olarak eğit

    # Kullanıcı tarafından seçilen modellere göre skip listesini güncelle
    global selected_models
    if "LinearSVM" not in selected_models:
        skip[0] = 1
    if "1L_NN" not in selected_models:
        skip[1] = 1
    if "2L_NN" not in selected_models:
        skip[2] = 1
    if "Gaussian_Naive_Bayes" not in selected_models:
        skip[3] = 1

    scores = []  # Doğruluk skorlarını tutmak için liste oluştur
    # Eğitim işlemi
    X, Y = read_data()
    assert (len(X) == len(Y))

    X, Y = shuffle(X, Y)

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, train_size=0.8)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    scores = []
    for idx, clf in tqdm(enumerate(classifiers), desc='Classifiers'):
        if not skip[idx]:  # Belirli modelleri atla
            clf.fit(X_train, Y_train)  # Modeli eğitme
            score = clf.score(X_test, Y_test)  # Test verisi üzerinde doğruluk skoru hesaplama
            scores.append(score)  # Skoru listeye ekle
            print(score)


            # Modeli kaydetme
            destination = f'models'
            if not os.path.exists(destination):
                os.makedirs(destination)

            location = f'models/{names[idx]}.sav'
            pickle.dump(clf, open(location, 'wb'))

    with open('models/report.txt', 'w') as fo:
        for score, name in zip(scores, names):
            fo.writelines(f'Score of {name}: {score}\n')


if __name__ == "__main__":
    train()