import numpy as np
import re

def przetwarzanie_tekstu(tekst):
    return re.sub(r'[^\w\s]', '', tekst).lower().split()

def utworz_macierz_terminow_dokumentow(dokumenty):
    terminy = list(set(slowo for dokument in dokumenty for slowo in dokument))
    indeks_terminow = {termin: idx for idx, termin in enumerate(terminy)}

    macierz = np.zeros((len(terminy), len(dokumenty)))
    for j, dokument in enumerate(dokumenty):
        for slowo in dokument:
            macierz[indeks_terminow[slowo], j] = 1
    return macierz, terminy


def redukcja_lsi(macierz, wektor_zapytania, wymiar):
    U, S, VT = np.linalg.svd(macierz, full_matrices=False)

    Uk = U[:, :wymiar]
    Sk = np.diag(S[:wymiar])
    VkT = VT[:wymiar, :]

    zapytanie_zredukowane = np.dot(np.linalg.inv(Sk), np.dot(Uk.T, wektor_zapytania))

    return np.dot(Sk, VkT), zapytanie_zredukowane


def podobienstwo_cos(macierz, wektor_zapytania):
    podobienstwa = []
    for wektor_dokumentu in macierz.T:
        norma_zapytania = np.linalg.norm(wektor_zapytania)
        norma_dokumentu = np.linalg.norm(wektor_dokumentu)
        podobienstwo = np.dot(wektor_dokumentu, wektor_zapytania) / (norma_zapytania * norma_dokumentu)
        podobienstwa.append(podobienstwo)
    return podobienstwa


def main():
    liczba_dokumentow = int(input())
    dokumenty = [przetwarzanie_tekstu(input()) for i in range(liczba_dokumentow)]
    zapytanie = przetwarzanie_tekstu(input())
    wymiar = int(input())

    macierz_terminow_dokumentow, terminy = utworz_macierz_terminow_dokumentow(dokumenty)

    wektor_zapytania = np.zeros(macierz_terminow_dokumentow.shape[0])
    for slowo in zapytanie:
        if slowo in terminy:
            wektor_zapytania[terminy.index(slowo)] = 1

    macierz_zredukowana, zapytanie_zredukowane = redukcja_lsi(macierz_terminow_dokumentow, wektor_zapytania, wymiar)

    podobienstwa = podobienstwo_cos(macierz_zredukowana, zapytanie_zredukowane)

    sformatowane_podobienstwa = [round(float(pod), 2) for pod in podobienstwa]
    print(sformatowane_podobienstwa)


if __name__ == "__main__":
    main()
