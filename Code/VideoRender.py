import multiprocessing
import numpy
import tqdm
import imageio
import matplotlib.cm as cm


# Settings

threads = 16
frame = 1800

input  = 'Data/Data{:04d}.npy'
output = 'Data/Frame{:04d}.png'

power = 0.2


# Code

def stats(i):
    im = numpy.load(input.format(i))
    return (im.min(), im.max())


def convert(i, minValue, maxValue):
    im = numpy.load(input.format(i))
    
    im = im.astype(float)
    im -= im.min() #minValue
    im /= im.max() #maxValue
    im = numpy.power(im, power)
    im = cm.inferno(im)
    im *= 255
    im = im.astype(numpy.uint8)
    
    imageio.imsave(output.format(i), im, compress_level = 3)


if __name__ == '__main__':
    minValue = numpy.inf
    maxValue = 0

    pool = multiprocessing.Pool(threads)

    for i in tqdm.tqdm(range((frame + threads - 1) // threads), desc = 'Scanning', ascii = True):
        start = i * threads
        stop  = min(start + threads, frame)

        m = numpy.array(pool.map(stats, numpy.arange(start, stop)))

        minValue = min(minValue, m.min())
        maxValue = max(minValue, m.max())

        start = start + threads

    print()
    print('Min value = {:d}'.format(minValue))
    print('Max value = {:d}'.format(maxValue))
    print()

    for i in tqdm.tqdm(range((frame + threads - 1) // threads), desc = 'Converting', ascii = True):
        start = i * threads
        stop  = min(start + threads, frame)

        parameter = zip(numpy.arange(start, stop), numpy.zeros(stop - start) + minValue, numpy.zeros(stop - start) + maxValue)
        
        pool.starmap(convert, parameter)

        start = start + threads
