import numpy as np  # NumPy kütüphanesi
import cv2 as cv  # OpenCV kütüphanesi
from scipy.ndimage import interpolation as inter  # SciPy'nin interpolation modülü
from PIL import Image as im  # PIL kütüphanesi


def binary_otsus(image, filter:int=1):
    """ Otsu'nun ikili (siyah-beyaz) resim dönüştürme yöntemini uygular.

    Args:
        image (numpy.ndarray): Giriş görüntü.
        filter (int): Görüntüyü yumuşatma (blurring) için kullanılan filtre boyutu. Varsayılan olarak 1.

    Returns:
        numpy.ndarray: Otsu'nun ikili dönüşüm sonucu elde edilen görüntü."""

    # Giriş resmi gri tonlamaya dönüştürme
    if len(image.shape) == 3:
        gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray_img = image

    # Otsu'nun ikili dönüşümü
    if filter != 0:
        blur = cv.GaussianBlur(gray_img, (3,3), 0)
        binary_img = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
    else:
        binary_img = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
    
    # Morphological Opening
    # kernel = np.ones((3,3),np.uint8)
    # clean_img = cv.morphologyEx(binary_img, cv.MORPH_OPEN, kernel)

    return binary_img


def find_score(arr, angle):

    """Belirtilen açıda döndürülmüş görüntünün skorunu hesaplar.

    Args:
        arr (numpy.ndarray): Giriş görüntü.
        angle (float): Açı.

    Returns:
        numpy.ndarray: Histogram.
        float: Skor."""
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score

def deskew(binary_img):
    """Görüntüyü doğrultma işlemi yapar.

    Args:
        binary_img (numpy.ndarray): İkili (siyah-beyaz) görüntü.

    Returns:
        numpy.ndarray: Doğrultulmuş görüntü."""
    
    ht, wd = binary_img.shape
    # _, binary_img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

    # pix = np.array(img.convert('1').getdata(), np.uint8)
    bin_img = (binary_img // 255.0)  # Görüntüyü ikili (siyah-beyaz) hale getirme

    delta = 0.1
    limit = 3
    angles = np.arange(-limit, limit+delta, delta)
    scores = []

     # Farklı açılarda skorları hesaplama
    for angle in angles:
        hist, score = find_score(bin_img, angle)
        scores.append(score)

    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    # print('Best angle: {}'.formate(best_angle))

    # Doğrultma işlemini gerçekleştirme
    data = inter.rotate(bin_img, best_angle, reshape=False, order=0)
    img = im.fromarray((255 * data).astype("uint8"))

    # img.save('skew_corrected.png')
    pix = np.array(img)
    return pix

def vexpand(gray_img, color:int):
    """Gri tonlamalı görüntüyü dikey olarak genişletir.

    Args:
        gray_img (numpy.ndarray): Gri tonlamalı giriş görüntüsü.
        color (int): Genişletilen bölgenin rengi (siyah için 0, beyaz için 1).

    Returns:
        numpy.ndarray: Genişletilmiş görüntü."""

    color = 1 if color > 0 else 0
    (h, w) = gray_img.shape[:2]
    space = np.ones((10, w)) * 255 * color

    return np.block([[space], [gray_img], [space]])


def hexpand(gray_img, color:int):
    """Gri tonlamalı görüntüyü yatay olarak genişletir.

    Args:
        gray_img (numpy.ndarray): Gri tonlamalı giriş görüntüsü.
        color (int): Genişletilen bölgenin rengi (siyah için 0, beyaz için 1).

    Returns:
        numpy.ndarray: Genişletilmiş görüntü."""

    color = 1 if color > 0 else 0
    (h, w) = gray_img.shape[:2]
    space = np.ones((h, 10)) * 255 * color

    return np.block([space, gray_img, space])

def valid(row, col, vis, word):
    """Verilen satır ve sütun indeksleri için geçerli bir hücre kontrolü yapar.

    Args:
        row (int): Satır indeksi.
        col (int): Sütun indeksi.
        vis (numpy.ndarray): Ziyaret edilen hücrelerin izdüşümü.
        word (numpy.ndarray): Kelimenin (görüntünün) piksel değerlerini içeren dizi.

    Returns:
        bool: Hücre geçerli ise True, değilse False döner."""

    return (row < vis.shape[0] and col < vis.shape[1] and row >= 0 and col >=0 and vis[row][col] == 0 and word[row][col] > 0)

def dfs(row, col, vis, word):
    """ Derin öncelikli arama (DFS) algoritması ile bağlı bileşenleri bulur.

    Args:
        row (int): Başlangıç satır indeksi.
        col (int): Başlangıç sütun indeksi.
        vis (numpy.ndarray): Ziyaret edilen hücrelerin izdüşümü.
        word (numpy.ndarray): Kelimenin (görüntünün) piksel değerlerini içeren dizi.

    Returns:
        None"""

    dX = [0,0,1,1,-1,-1,1,-1]
    dY = [1,-1,0,1,0,-1,-1,1]
    vis[row][col] += 1
    for i in range(8):
        if(valid(row+dX[i],col+dY[i],vis, word)):
            dfs(row+dX[i], col+dY[i], vis, word)
    return
