import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def printi(img, img_title="image"):
    """ Pomocnicza funkcja do wypisania informacji o obrazie. """
    print(
        f"{img_title}, wymiary: {img.shape}, typ danych: {img.dtype}, wartości: {img.min()} - {img.max()}"
    )


def cv_imshow(img, img_title="image"):
    """
    Funkcja do wyświetlania obrazu w wykorzystaniem okna OpenCV.
    Wykonywane jest przeskalowanie obrazu z rzeczywistymi lub 16-bitowymi całkowitoliczbowymi wartościami pikseli,
    żeby jedną funkcją wyświetlać obrazy różnych typów.
    """
    if (img.dtype == np.float32) or (img.dtype == np.float64):
        img_ = img / 255
    elif img.dtype == np.int16:
        img_ = img * 128
    else:
        img_ = img
    cv2.imshow(img_title, img_)
    cv2.waitKey(1)


""" Wczytanie obrazów z plików """
image_mono = cv2.imread("boat_mono.png", cv2.IMREAD_UNCHANGED)
image_col = cv2.imread("boat_col.png", cv2.IMREAD_UNCHANGED)

### DLA OBRAZU MONOCHROMATYCZNEGO ###

printi(image_mono, "image_mono")

""" 
Obliczenie przepływności. os.stat() podaje rozmiar pliku w bajtach, a potrzebny jest w bitach, dlatego należy pomnożyć * 8
"""
bitrate = (
    8 * os.stat("boat_mono.png").st_size / (image_mono.shape[0] * image_mono.shape[1])
)
print(f"bitrate: {bitrate:.4f}")


def calc_entropy(hist):
    pdf = hist / hist.sum()
    entropy = -sum([x * np.log2(x) for x in pdf if x != 0])
    return entropy


# Wyznacznenie histogramu obrazu oryginalnego
hist_image_mono = cv2.calcHist([image_mono], [0], None, [256], [0, 256])
hist_image_mono = hist_image_mono.flatten()

### Sprawdzenie czy suma wartości histogramu równa jest liczbie piskeli w obrazie ###
print(f"suma wartosci histogramu: {hist_image_mono.sum()}")
print(f"liczba pikseli w obrazie: {image_mono.shape[0] * image_mono.shape[1]}")

""" Obliczenie entropii obrazu oryginalnego """
H_image_mono = calc_entropy(hist_image_mono)
print(f"H(image_mono) = {H_image_mono:.4f}")

""" Wyznaczenie obrazu różnicowego """
img_tmp1 = image_mono[:, 1:]
img_tmp2 = image_mono[:, :-1]
image_mono_hdiff = cv2.addWeighted(img_tmp1, 1, img_tmp2, -1, 0, dtype=cv2.CV_16S)
image_mono_hdiff_0 = cv2.addWeighted(image_mono[:, 0], 1, 0, 0, -127, dtype=cv2.CV_16S)
image_mono_hdiff = np.hstack((image_mono_hdiff_0, image_mono_hdiff))
printi(image_mono_hdiff, "image_mono_hdiff")
cv_imshow(image_mono_hdiff, "image_mono_hdiff")


# Wyznaczenie histogramu obrazu różnicowego
image_tmp = (image_mono_hdiff + 255).astype(np.uint16)
hist_mono_hdiff = cv2.calcHist([image_tmp], [0], None, [511], [0, 511]).flatten()

# Wyświetlenie histogramu obrazu oryginalnego i różnicowego
plt.figure()
plt.plot(hist_image_mono, color="blue")
plt.title("hist_image_mono")
plt.xlim([0, 255])
plt.figure()
plt.plot(np.arange(-255, 256, 1), hist_mono_hdiff, color="red")
plt.title("hist_mono_hdiff")
plt.xlim([-255, 255])
plt.show()
cv2.waitKey(0)

# Wyznaczenie entropii obrazu różnicowego
H_mono_hdiff = calc_entropy(hist_mono_hdiff)
print(f"H(hdiff) = {H_mono_hdiff:.4f}")


def dwt(img):
    maskL = np.array(
        [
            0.02674875741080976,
            -0.01686411844287795,
            -0.07822326652898785,
            0.2668641184428723,
            0.6029490182363579,
            0.2668641184428723,
            -0.07822326652898785,
            -0.01686411844287795,
            0.02674875741080976,
        ]
    )
    maskH = np.array(
        [
            0.09127176311424948,
            -0.05754352622849957,
            -0.5912717631142470,
            1.115087052456994,
            -0.5912717631142470,
            -0.05754352622849957,
            0.09127176311424948,
        ]
    )

    bandLL = cv2.sepFilter2D(img, -1, maskL, maskL)[::2, ::2]
    bandLH = cv2.sepFilter2D(img, cv2.CV_16S, maskL, maskH)[::2, ::2]
    bandHL = cv2.sepFilter2D(img, cv2.CV_16S, maskH, maskL)[::2, ::2]
    bandHH = cv2.sepFilter2D(img, cv2.CV_16S, maskH, maskH)[::2, ::2]

    return bandLL, bandLH, bandHL, bandHH


# Wyznaczenie współczynników DWT
ll, lh, hl, hh = dwt(image_mono)
# Wyświetlenie pasm
printi(ll, "LL")
printi(lh, "LH")
printi(hl, "HL")
printi(hh, "HH")
cv_imshow(ll, "LL2")
cv_imshow(lh, "LH2")
cv_imshow(hl, "HL2")
cv_imshow(hh, "HH2")

# Wyznaczenie histogramów dla pasm
hist_ll = cv2.calcHist([ll], [0], None, [256], [0, 256]).flatten()
hist_lh = cv2.calcHist(
    [(lh + 255).astype(np.uint16)], [0], None, [511], [0, 511]
).flatten()
hist_hl = cv2.calcHist(
    [(hl + 255).astype(np.uint16)], [0], None, [511], [0, 511]
).flatten()
hist_hh = cv2.calcHist(
    [(hh + 255).astype(np.uint16)], [0], None, [511], [0, 511]
).flatten()

# Obliczenie entropii
H_ll = calc_entropy(hist_ll)
H_lh = calc_entropy(hist_lh)
H_hl = calc_entropy(hist_hl)
H_hh = calc_entropy(hist_hh)

print(
    f"H(LL) = {H_ll:.4f} \nH(LH) = {H_lh:.4f} \nH(HL) = {H_hl:.4f} \nH(HH) = {H_hh:.4f} \nH_śr = {(H_ll + H_lh + H_hl + H_hh) / 4:.4f}"
)

fig = plt.figure()
fig.set_figheight(fig.get_figheight() * 2)  ### zwiększenie rozmiarów okna
fig.set_figwidth(fig.get_figwidth() * 2)
plt.subplot(2, 2, 1)
plt.plot(hist_ll, color="blue")
plt.title("hist_ll")
plt.xlim([0, 255])
plt.subplot(2, 2, 3)
plt.plot(np.arange(-255, 256, 1), hist_lh, color="red")
plt.title("hist_lh")
plt.xlim([-255, 255])
plt.subplot(2, 2, 2)
plt.plot(np.arange(-255, 256, 1), hist_hl, color="red")
plt.title("hist_hl")
plt.xlim([-255, 255])
plt.subplot(2, 2, 4)
plt.plot(np.arange(-255, 256, 1), hist_hh, color="red")
plt.title("hist_hh")
plt.xlim([-255, 255])
plt.show()
cv2.waitKey(0)


### DLA OBRAZU BARWNEGO ###

printi(image_col, "image_col")

image_R = image_col[:, :, 2]
image_G = image_col[:, :, 1]
image_B = image_col[:, :, 0]

# Wyznaczenie histogramów dla składowych RGB
hist_R = cv2.calcHist([image_R], [0], None, [256], [0, 256]).flatten()
hist_G = cv2.calcHist([image_G], [0], None, [256], [0, 256]).flatten()
hist_B = cv2.calcHist([image_B], [0], None, [256], [0, 256]).flatten()

# Obliczenie entropii dla składowych RGB
H_R = calc_entropy(hist_R)
H_G = calc_entropy(hist_G)
H_B = calc_entropy(hist_B)
print(
    f"H(R) = {H_R:.4f} \nH(G) = {H_G:.4f} \nH(B) = {H_B:.4f} \nH_śr = {(H_R + H_G + H_B) / 3:.4f}"
)

# Konwersja z RGB do YUV
image_YUV = cv2.cvtColor(image_col, cv2.COLOR_BGR2YUV)

# Wyznaczenie histogramów dla składowych YUV
hist_Y = cv2.calcHist([image_YUV[:, :, 0]], [0], None, [256], [0, 256]).flatten()
hist_U = cv2.calcHist([image_YUV[:, :, 1]], [0], None, [256], [0, 256]).flatten()
hist_V = cv2.calcHist([image_YUV[:, :, 2]], [0], None, [256], [0, 256]).flatten()

# Wyznaczenie entropii dla składowych YUV
H_Y = calc_entropy(hist_Y)
H_U = calc_entropy(hist_U)
H_V = calc_entropy(hist_V)

print(
    f"H(Y) = {H_Y:.4f} \nH(U) = {H_U:.4f} \nH(V) = {H_V:.4f} \nH_śr = {(H_Y + H_U + H_V) / 3:.4f}"
)


cv_imshow(image_R, "image_R")
cv_imshow(image_G, "image_G")
cv_imshow(image_B, "image_B")
plt.figure()
plt.plot(hist_R, color="red")
plt.plot(hist_G, color="green")
plt.plot(hist_B, color="blue")
plt.title("hist RGB")
plt.xlim([0, 255])
plt.show()
cv2.waitKey(0)

cv_imshow(image_YUV[:, :, 0], "image_Y")
cv_imshow(image_YUV[:, :, 1], "image_U")
cv_imshow(image_YUV[:, :, 2], "image_V")
plt.figure()
plt.plot(hist_Y, color="gray")
plt.plot(hist_U, color="red")
plt.plot(hist_V, color="blue")
plt.title("hist YUV")
plt.xlim([0, 255])
plt.show()
cv2.waitKey(0)

""" Wyznaczanie charakterystyki R-D """


def calc_mse_psnr(img1, img2):
    """ Funkcja obliczająca MSE i PSNR dla podanych obrazów, zakładana wartość pikseli z przedziału [0, 255]. """

    imax = 255.0 ** 2
    mse = ((img1.astype(np.float64) - img2) ** 2).sum() / img1.size
    psnr = 10.0 * np.log10(imax / mse)
    return mse, psnr


xx = []  ### tablica na wartości osi X -> bitrate
ym = []  ### tablica na wartości osi Y dla MSE
yp = []  ### tablica na wartości osi Y dla PSNR

for quality in [
    100,
    95,
    90,
    70,
    60,
    50,
    30,
    20,
    10,
]:
    out_file_name = f"out_image_q{quality:03d}.jpg"
    """ Zapis do pliku w formacie .jpg z ustaloną 'jakością' """
    cv2.imwrite(out_file_name, image_col, (cv2.IMWRITE_JPEG_QUALITY, quality))
    """ Odczyt skompresowanego obrazu, policzenie bitrate'u i PSNR """
    image_compressed = cv2.imread(out_file_name, cv2.IMREAD_UNCHANGED)
    bitrate = (
        8 * os.stat(out_file_name).st_size / (image_col.shape[0] * image_col.shape[1])
    )
    mse, psnr = calc_mse_psnr(image_col, image_compressed)
    xx.append(bitrate)
    ym.append(mse)
    yp.append(psnr)

""" Narysowanie wykresów """
fig = plt.figure()
fig.set_figwidth(fig.get_figwidth() * 2)
plt.suptitle("Charakterystyki R-D")
plt.subplot(1, 2, 1)
plt.plot(xx, ym, "-.")
plt.title("MSE(R)")
plt.xlabel("bitrate")
plt.ylabel("MSE", labelpad=0)
plt.subplot(1, 2, 2)
plt.plot(xx, yp, "-o")
plt.title("PSNR(R)")
plt.xlabel("bitrate")
plt.ylabel("PSNR [dB]", labelpad=0)
plt.show()

# Wyliczenie przepływności bitowej dla obrazu barwnego skompresowanego koderem PNG
bitrate_col_png = (
    8 * os.stat("boat_col.png").st_size / (image_col.shape[0] * image_col.shape[1])
)
print(f"bitrate col png: {bitrate_col_png:.4f}")

# Wyliczenie przepływności bitowej dla obrazu barwnego skompresowanego koderem JPEG
bitrate_jpg = (
    8
    * os.stat("out_image_q100.jpg").st_size
    / (image_col.shape[0] * image_col.shape[1])
)
print(f"bitrate jpeg: {bitrate_jpg:.4f}")

cv2.waitKey(0)
cv2.destroyAllWindows()
