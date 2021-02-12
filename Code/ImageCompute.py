import multiprocessing
import os.path as path
import copy
import numpy
import time
import matplotlib.cm as cm
import imageio


# Settings

threads = 16

size = 5_000_000
skip = 10
iteration = 100

angle = (8 / 9) * 2.0 * numpy.pi
mid = [0, 0, 0]
r = 8
perspective = 1
viewport = 4
power = 0.2

outputData  = 'Data/image.npy'
outputImage = 'Data/image.png'
resolution = [3840, 2160]


# Functions

def rotateX(angle):
    m = numpy.zeros((3, 3))
    
    m[0, 0] = 1.
    m[1, 1] = +numpy.cos(angle)
    m[1, 2] = -numpy.sin(angle)
    m[2, 1] = +numpy.sin(angle)
    m[2, 2] = +numpy.cos(angle)
    
    return m


def rotateY(angle):
    m = numpy.zeros((3, 3))
    
    m[1, 1] = 1.
    m[0, 0] = +numpy.cos(angle)
    m[0, 2] = +numpy.sin(angle)
    m[2, 0] = -numpy.sin(angle)
    m[2, 2] = +numpy.cos(angle)
    
    return m


def rotateZ(angle):
    m = numpy.zeros((3, 3))
    
    m[2, 2] = 1.
    m[0, 0] = +numpy.cos(angle)
    m[0, 1] = -numpy.sin(angle)
    m[1, 0] = +numpy.sin(angle)
    m[1, 1] = +numpy.cos(angle)
    
    return m


def attractor(input, parameter):
    output = numpy.zeros(input.shape)

    output[:, 0] = input[:, 2] * numpy.sin(parameter[0] + input[:, 1] * input[:, 0]) + input[:, 1] * numpy.cos(parameter[3] + input[:, 2] * input[:, 0])
    output[:, 1] = input[:, 0] * numpy.sin(parameter[1] + input[:, 2] * input[:, 1]) + input[:, 2] * numpy.cos(parameter[4] + input[:, 0] * input[:, 1])
    output[:, 2] = input[:, 1] * numpy.sin(parameter[2] + input[:, 0] * input[:, 2]) + input[:, 0] * numpy.cos(parameter[5] + input[:, 1] * input[:, 2])

    return output


def probe(size, iteration, parameter):
    pointCloud = numpy.random.normal(0, 1, (size, 3))
    d = numpy.sqrt((pointCloud ** 2).sum(axis = 1))
    pointCloud[:, 0] /= d
    pointCloud[:, 1] /= d
    pointCloud[:, 2] /= d

    for i in range(iteration):
        pointCloud = attractor(pointCloud, parameter)

    xMid = numpy.array([pointCloud[:, 0].min(), pointCloud[:, 0].max()]).mean()
    yMid = numpy.array([pointCloud[:, 1].min(), pointCloud[:, 1].max()]).mean()
    zMid = numpy.array([pointCloud[:, 2].min(), pointCloud[:, 2].max()]).mean()
    mid = numpy.array([xMid, yMid, zMid])

    r = numpy.sqrt(numpy.percentile((((pointCloud - mid) ** 2.0).sum(axis = 1)), 99.9))

    return pointCloud, mid, r


def render(start, iteration, parameter, resolution, mid, r, transformationMatrix = None, perspective = 1, viewport = 1):
    im = numpy.zeros((resolution[1], resolution[0]))
    pointCloud = start

    for i in range(iteration):
        pointCloud = attractor(pointCloud, parameter)

        temporal = copy.deepcopy(pointCloud)
        temporal -= mid
        temporal /= (r * 2)

        if transformationMatrix is not None:
            temporal = temporal.dot(transformationMatrix)
            
        temporal[:, 0] = temporal[:, 0] / (1. + (.5 + temporal[:, 2]) * perspective)
        temporal[:, 1] = temporal[:, 1] / (1. + (.5 + temporal[:, 2]) * perspective)

        temporal[:, 0] *= viewport
        temporal[:, 1] *= viewport

        factor = .5 / min(resolution)
    
        xRange = [-(resolution[0] * factor), resolution[0] * factor]
        yRange = [-(resolution[1] * factor), resolution[1] * factor]

        im += numpy.histogram2d(temporal[:, 0], temporal[:, 1], bins = resolution, range = [xRange, yRange])[0].T

    return pointCloud, im


def compute(angle, mid, r, viewport):
    cycle = 2. * numpy.pi

    a = numpy.sin(angle) * .2 + numpy.sin(1 * angle + cycle * (0 / 6)) * .3  + numpy.sin(5 * angle + cycle * (0 / 6)) * .15
    b = numpy.sin(angle) * .2 + numpy.sin(1 * angle + cycle * (1 / 6)) * .3  + numpy.sin(5 * angle + cycle * (1 / 6)) * .15
    c = numpy.sin(angle) * .2 + numpy.sin(1 * angle + cycle * (2 / 6)) * .3  + numpy.sin(5 * angle + cycle * (2 / 6)) * .15 
    d = numpy.sin(angle) * .2 + numpy.sin(1 * angle + cycle * (3 / 6)) * .3  + numpy.sin(5 * angle + cycle * (3 / 6)) * .15 
    e = numpy.sin(angle) * .2 + numpy.sin(1 * angle + cycle * (4 / 6)) * .3  + numpy.sin(5 * angle + cycle * (4 / 6)) * .15
    f = numpy.sin(angle) * .2 + numpy.sin(1 * angle + cycle * (5 / 6)) * .3  + numpy.sin(5 * angle + cycle * (5 / 6)) * .15

    pointCloud, _, _ = probe(size, skip, [a, b, c, d, e, f])

    transformationMatrix = numpy.matmul(rotateX(angle * 2), rotateY(angle * 3))

    _, im = render(pointCloud, iteration, [a, b, c, d, e, f], resolution, mid, r, transformationMatrix, perspective, viewport)

    return im


if __name__ == '__main__':
    
    final = numpy.zeros((resolution[1], resolution[0]))
    
    if path.exists(outputData):
        final = numpy.load(outputData)

    pool = multiprocessing.Pool(threads)

    parameter = []

    for i in range(threads):
        parameter.append([angle, mid, r, viewport])

    i = 0
    t0 = time.time()

    while True:
        i += 1

        im = numpy.array(pool.starmap(compute, parameter)).sum(axis = 0)

        final += im

        numpy.save(outputData, final)

        im = numpy.array(final)
        im -= im.min()
        im /= im.max()
        im = numpy.power(im, power)
        im = cm.inferno(im)
        im *= 255
        im = im.astype(numpy.uint8)

        imageio.imsave(outputImage, im, compress_level = 3)

        print('[{:d}] Points: {:.1f} G'.format(int(time.time() - t0), final.sum() / 1_000_000_000))

