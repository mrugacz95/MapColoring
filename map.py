#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as pyplot
import szablon as col


def simpleShadow(heights, colorData):
    ret = np.zeros((heights.shape[0], heights.shape[1], 3), dtype=np.uint8)
    for index in np.ndindex(heights.shape):
        val = colorData[index]
        x, y = index
        if heights[x][y - 1] < heights[x][y]:
            ret[index] = val / 2
        else:
            ret[index] = val
    return ret


def tableToRGB(array, gradient=col.gradient_hsv_known):
    data = np.zeros((array.shape[0], array.shape[1], 3), dtype=np.uint8)
    for index in np.ndindex(array.shape):
        (r, g, b) = (x * 255 for x in gradient(array[index]))
        data[index] = [r, g, b]
    return data


def normalizeVec(a):
    a = np.array(a)
    return a / np.linalg.norm(a)


def subVec(a, b):
    v = [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    return v


def calcVector(a, b, c):
    vec = np.cross(subVec(b, a), subVec(c, a))
    return vec / length(vec)


def length(vec):
    return np.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)


def calcShadow(heights, sunPos):
    size = heights.shape[0]
    ret = np.zeros(heights.shape, dtype=float)
    sunPos = normalizeVec(sunPos)
    for index in np.ndindex(heights.shape):
        x, y = index
        val = calcVector([0, 0, heights[x][(y) % size]], [1, 0, heights[(x + 1) % size][y]],
                         [1, 1, heights[(x + 1) % size][(y + 1) % size]])
        angle = np.dot(val, sunPos)
        ret[index] = 1 - angle
    ret = normalize(ret)
    return ret


def normalize(array):
    max = array.max()
    min = array.min()
    norm = lambda x: (x - min) / (max - min)
    return norm(array)


def gradient_hsvShadows(h, s, v):
    return col.hsv2rgb(1 / 3 - h / 3, s, v)


def addShadow(heights, shadows):
    data = np.zeros((heights.shape[0], heights.shape[1], 3), dtype=np.uint8)
    for index in np.ndindex(heights.shape):
        (r, g, b) = (x * 255 for x in
                     gradient_hsvShadows(heights[index], 1, shadows[index]))
        data[index] = [r, g, b]
    return data


def main():
    sunPos = [10, 10, 0.5]
    with open('big.dem') as file:
        subplots = (2, 3)
        heights = np.loadtxt(file, skiprows=1, dtype=float)

        pyplot.subplot(subplots[0], subplots[1], 1)
        pyplot.imshow(heights)  # RawData

        heights = normalize(heights)

        pyplot.subplot(subplots[0], subplots[1], 2)
        gradientData = tableToRGB(heights, col.gradient_rgb_gbr_full)
        pyplot.imshow(gradientData)  # Without shadow

        pyplot.subplot(subplots[0], subplots[1], 3)
        colorData = tableToRGB(heights)
        pyplot.imshow(colorData)  # Without shadow

        pyplot.subplot(subplots[0], subplots[1], 4)
        withSimpleShadow = simpleShadow(heights, colorData)
        pyplot.imshow(withSimpleShadow)  # Simple shadow

        pyplot.subplot(subplots[0], subplots[1], 5)
        shadow = calcShadow(heights, sunPos)
        tableToRGB(shadow, col.gradient_rgb_bw)
        pyplot.imshow(tableToRGB(shadow, col.gradient_rgb_bw))  # Only shadows

        pyplot.subplot(subplots[0], subplots[1], 6)
        final = addShadow(heights, shadow)
        pyplot.imshow(final)  # Final
        pyplot.show()


if __name__ == '__main__':
    main()
