import numpy as np  # NumPy kütüphanesi
import cv2 as cv  # OpenCV kütüphanesi
import matplotlib.pyplot as plt  # Matplotlib kütüphanesi
from preprocessing import binary_otsus, deskew  # Ön işleme fonksiyonları
from utilities import *  # Yardımcı fonksiyonlar
import pandas as pd  # Pandas kütüphanesi
from PIL import Image  # PIL kütüphanesi
import glob  # Dosya yollarını almak için kullanılan modül
import os  # İşletim sistemi işlemleri için kullanılan modül
from sklearn.model_selection import train_test_split  # Veri kümesini ayırmak için
from sklearn.svm import SVC  # SVM sınıflandırıcı
from sklearn.metrics import classification_report, confusion_matrix  # Performans metrikleri
import warnings  # Uyarıları kontrol etmek için

# %matplotlib inline
def whiteBlackRatio(img):
    # Beyaz ve siyah piksel oranını hesaplayan fonksiyon
    h = img.shape[0]
    w = img.shape[1]
    blackCount = 1
    whiteCount = 0
    for y in range(0, h):
        for x in range(0, w):
            if img[y, x] == 0:
                blackCount += 1
            else:
                whiteCount += 1
    return whiteCount / blackCount

# Siyah piksel sayısını hesaplayan fonksiyon
def blackPixelsCount(img):
    blackCount = 1  # Sıfıra bölme hatasını önlemek için 1 ile başlatılır
    h = img.shape[0]
    w = img.shape[1]
    for y in range(0, h):
        for x in range(0, w):
            if img[y, x] == 0:
                blackCount += 1
    return blackCount

# Yatay geçiş sayısını hesaplayan fonksiyon
def horizontalTransitions(img):
    h = img.shape[0]
    w = img.shape[1]
    maximum = 0
    for y in range(0, h):
        prev = img[y, 0]
        transitions = 0
        for x in range(1, w):
            if img[y, x] != prev:
                transitions += 1
                prev = img[y, x]
        maximum = max(maximum, transitions)
    return maximum

# Dikey geçiş sayısını hesaplayan fonksiyon
def verticalTransitions(img):
    h = img.shape[0]
    w = img.shape[1]
    maximum = 0
    for x in range(0, w):
        prev = img[0, x]
        transitions = 0
        for y in range(1, h):
            if img[y, x] != prev:
                transitions += 1
                prev = img[y, x]
        maximum = max(maximum, transitions)
    return maximum

# Histogram ve ağırlık merkezini hesaplayan fonksiyon
def histogramAndCenterOfMass(img):
    h = img.shape[0]
    w = img.shape[1]
    histogram = []
    sumX = 0
    sumY = 0
    num = 0
    for x in range(0, w):
        localHist = 0
        for y in range(0, h):
            if img[y, x] == 0:
                sumX += x
                sumY += y
                num += 1
                localHist += 1
        histogram.append(localHist)
    return sumX / num, sumY / num, histogram

# Özellikleri çıkaran fonksiyon
def getFeatures(img):
    x, y = img.shape
    featuresList = []
    # Yükseklik/genişlik oranı
    featuresList.append(y / x)
    # Siyah ve beyaz piksel oranı
    featuresList.append(whiteBlackRatio(img))
    # Yatay ve dikey geçiş sayıları
    featuresList.append(horizontalTransitions(img))
    featuresList.append(verticalTransitions(img))

    # İmgenin dörtte birlerine ayrılması
    topLeft = img[0:y // 2, 0:x // 2]
    topRight = img[0:y // 2, x // 2:x]
    bottomLeft = img[y // 2:y, 0:x // 2]
    bottomRight = img[y // 2:y, x // 2:x]

    # Her dörtte birdeki beyazdan siyaha oranı
    featuresList.append(whiteBlackRatio(topLeft))
    featuresList.append(whiteBlackRatio(topRight))
    featuresList.append(whiteBlackRatio(bottomLeft))
    featuresList.append(whiteBlackRatio(bottomRight))

    # Diğer 6 özellik
    topLeftCount = blackPixelsCount(topLeft)
    topRightCount = blackPixelsCount(topRight)
    bottomLeftCount = blackPixelsCount(bottomLeft)
    bottomRightCount = blackPixelsCount(bottomRight)

    featuresList.append(topLeftCount / topRightCount)
    featuresList.append(bottomLeftCount / bottomRightCount)
    featuresList.append(topLeftCount / bottomLeftCount)
    featuresList.append(topRightCount / bottomRightCount)
    featuresList.append(topLeftCount / bottomRightCount)
    featuresList.append(topRightCount / bottomLeftCount)

    # Kütle merkezi ve yatay histogram
    xCenter, yCenter, xHistogram = histogramAndCenterOfMass(img)
    featuresList.append(xCenter)
    featuresList.append(yCenter)

    return featuresList

# Alt dizinleri almak için kullanılan fonksiyon
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

# Tüm dosyaları almak için kullanılan fonksiyon
def getListOfFiles(dirName):
    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles

# Eğitim ve sınıflandırma işlemlerini yürüten fonksiyon
def trainAndClassify(data, classes):
    X_train, X_test, y_train, y_test = train_test_split(data, classes, test_size=0.20)  # Veriyi ayırma
    svclassifier = SVC(kernel='rbf', gamma=0.005, C=1000)  # RBF çekirdekli SVM sınıflandırıcı
    svclassifier.fit(X_train, y_train)  # Modeli eğitme
    y_pred = svclassifier.predict(X_test)  # Test verilerini kullanarak tahmin yapma
    print(confusion_matrix(y_test, y_pred))  # Confusion matrixi ekrana yazdırma
    print(classification_report(y_test, y_pred))  # Sınıflandırma raporunu ekrana yazdırma

# Kenar boşluklarını kaldıran fonksiyon
def removeMargins(img):
    th, threshed = cv.threshold(img, 245, 255, cv.THRESH_BINARY_INV)  # Eşikleme işlemi
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
    morphed = cv.morphologyEx(threshed, cv.MORPH_CLOSE, kernel)  # Morfolojik operasyonlar
    cnts = cv.findContours(morphed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]  # Contour bulma
    cnt = sorted(cnts, key=cv.contourArea)[-1]  # En büyük contouru bulma
    x, y, w, h = cv.boundingRect(cnt)  # Bounding box koordinatlarını alma
    dst = img[y:y + h, x:x + w]  # Kenar boşluklarını kaldırma
    return dst

# Ana fonksiyon
def main():
    data = np.array([])  # Özellikleri tutmak için dizi
    classes = np.array([])  # Sınıfları tutmak için dizi
    directory = '../LettersDataset'  # Veri seti dizini
    chars = get_immediate_subdirectories(directory)  # Alt dizinler (karakterler)
    count = 0
    numOfFeatures = 16  # Kullanılan özellik sayısı
    charPositions = ['Beginning', 'End', 'Isolated', 'Middle']  # Karakter pozisyonları
    for char in chars:
        for position in charPositions:
            if os.path.isdir(directory + '/' + char + '/' + position) == True:
                listOfFiles = getListOfFiles(directory + '/' + char + '/' + position)  # Dosya listesi
                for filename in listOfFiles:
                    img = cv.imread(filename)  # Resmi okuma
                    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Gri tonlamaya çevirme
                    cropped = removeMargins(gray_img)  # Kenar boşluklarını kaldırma
                    binary_img = binary_otsus(cropped, 0)  # Binary görüntüleme
                    features = getFeatures(binary_img)  # Özellikleri çıkarma
                    data = np.append(data, features)  # Özellikleri diziye ekleme
                    classes = np.append(classes, char + position)  # Sınıfları diziye ekleme
                    count += 1
    
    data = np.reshape(data, (count, numOfFeatures))  # Veriyi yeniden şekillendirme
    trainAndClassify(data, classes)  # Eğitim ve sınıflandırma işlemlerini yürütme

main()  # Ana fonksiyonu çalıştırma
