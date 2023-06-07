from django.shortcuts import render

import functools, math
import cv2
import numpy as np
import json

f = open("children.json")
data = json.load(f)
f.close()


# Create your views here.
def index(request):
    Data = {"imgex": data[1]["link"]}
    return render(request, "pages/base.html", Data)


def output(request):
    if request.method == "POST" and request.FILES["myfile"]:
        f = request.FILES["myfile"]
        with open("home/static/img/upload/" + f.name, "wb+") as destination:
            for chunk in f.chunks():
                destination.write(chunk)

        img = cv2.imread("home/static/img/upload/" + f.name, cv2.IMREAD_GRAYSCALE)
        his = histogramOfGradients(img)
        his_norm = normalize(his)
        hog = his_norm.flatten()
        print(len(hog))
        for i in hog:
            print(i, end=" ")
        distance = {}
        for i in data:
            distance[i["link"]] = euclid(hog, i["features"])
        distance = sorted(distance.items(), key=lambda item: item[1])
        result = []
        for i in range(5):
            print(distance[i][0], distance[i][1])
            result.append(distance[i][0])
        Data = {"output": result, "input": f.name}
        return render(request, "pages/home.html", Data)


def test(request):
    if request.method == "POST" and request.FILES["myfile"]:
        f = request.FILES["myfile"]
        with open("home/static/img/upload/" + f.name, "wb+") as destination:
            for chunk in f.chunks():
                destination.write(chunk)
    return render(request, "pages/home.html")


def euclid(vt1, vt2):
    s = 0
    for i in range(len(vt1)):
        s += (vt1[i] - vt2[i]) ** 2
    return math.sqrt(s)


def compute(receptiveField, kernel):
    result = 0
    for i in range(len(kernel)):
        for j in range(len(kernel)):
            result += receptiveField[i][j] * kernel[i][j]
    return result


def convolution(matrix, kernel):
    featureMap = []
    for i in range(1, len(matrix) - 1):
        x = []
        for j in range(1, len(matrix[0]) - 1):
            x.append(compute(matrix[i - 1 : i + 2, j - 1 : j + 2], kernel))
        featureMap.append(x)
    return featureMap


def gradients(img):
    img = np.pad(img, pad_width=1)
    sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    Gx = convolution(img, sobel_x)
    Gy = convolution(img, sobel_y)
    direction, magnitude = [], []
    for i in range(len(Gx)):
        d, m = [], []
        for j in range(len(Gx[0])):
            m.append(round(math.sqrt(Gx[i][j] ** 2 + Gy[i][j] ** 2)))
            d.append(
                round(math.atan(Gy[i][j] / Gx[i][j]) * 180 / math.pi)
                if Gx[i][j] != 0
                else 90
            )
        direction.append(d)
        magnitude.append(m)
    return np.array(direction), np.array(magnitude)


def unitHistogram(direction, magnitude):
    histogram = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(8):
        for j in range(8):
            x = direction[i][j] // 20
            if direction[i][j] % 20 == 0:
                histogram[x] += magnitude[i][j]
            else:
                k = round((direction[i][j] - x * 20) * magnitude[i][j] / 20)
                histogram[x] += magnitude[i][j] - k
                histogram[x + 1] += k
    histogram[0] += histogram[-1]
    return histogram[:-1]


def histogramOfGradients(img):
    direction, magnitude = gradients(img)
    result = []
    for i in range(0, len(img), 8):
        row = []
        for j in range(0, len(img[0]), 8):
            row.append(
                unitHistogram(
                    direction[i : i + 8, j : j + 8], magnitude[i : i + 8, j : j + 8]
                )
            )
        result.append(row)
    return np.array(result)


def norm2(arr):
    norm = 0
    for i in arr:
        norm += i**2
    if norm == 0:
        return arr
    return np.round(np.divide(arr, math.sqrt(norm)), decimals=5)


def normalize(matrix):
    result = []
    for i in range(len(matrix) - 1):
        row = []
        for j in range(len(matrix[0]) - 1):
            m = np.array(matrix[i : i + 2, j : j + 2]).flatten()
            row.append(norm2(m))
        result.append(row)
    return np.array(result)
