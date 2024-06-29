import numpy as np  # NumPy kütüphanesi
import cv2 as cv  # OpenCV kütüphanesi


def save_image(img, folder, title):
    """Verilen görüntüyü belirtilen klasöre ve dosya adıyla kaydeden fonksiyon."""
    cv.imwrite(f'./{folder}/{title}.png', img)


def projection(gray_img, axis: str = 'horizontal'):
    """Gri tonlamalı bir görüntünün yatay veya dikey iz düşümünü hesaplayan fonksiyon."""

    if axis == 'horizontal':  # Yatay iz düşüm hesaplama
        projection_bins = np.sum(gray_img, 1).astype('int32')
    elif axis == 'vertical':  # Dikey iz düşüm hesaplama
        projection_bins = np.sum(gray_img, 0).astype('int32')

    return projection_bins  # İz düşüm verilerini döndürme
