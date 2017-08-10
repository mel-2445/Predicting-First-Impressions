def addConstantOcclusion(originalImage, space):
    from numpy import uint8
    from random import randint
    h,w = originalImage.shape[:2]
    x2,y2,x1,y1,pct_area = 0,0,0,0,0
    max_area = space['occl_min_area'] + space['occl_area_distance']
    while x2 == x1 or y2 == y1 or pct_area < space['occl_min_area'] or pct_area > max_area:
        x1,y1 = (float(randint(0,w)), float(randint(0,h)))
        x2,y2 = (float(randint(0,w)), float(randint(0,h)))
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        pct_area = ((x2-x1)*(y2-y1))/(w*h)

    originalImage[int(y1):int(y2), int(x1):int(x2)] = uint8(0)
    return originalImage


def randomNoise(originalImage, space):
    import numpy as np
    from random import uniform
    h,w = originalImage.shape[:2]
    noise_freq = uniform(space['noise_min_freq_pct'], space['noise_max_freq_pct']) / 100.
    prob_mask = np.random.uniform(0, 1, (h,w))
    originalImage[np.where(prob_mask < noise_freq)] = 0
    return originalImage


def changeBrightness(originalImage, space):
    from random import randint
    from numpy import uint8, where, zeros_like, float64
    brighten_min = int(space['brighten_extent_min'])
    brighten_max = int(space['brighten_extent_max'])
    amountToBrighten = randint(brighten_min, brighten_max)
    newImage = float64(originalImage)
    if amountToBrighten > 0:
        newImage[where(newImage + amountToBrighten < 255)] += float64(amountToBrighten)
        newImage[newImage + amountToBrighten >= 255] = 255.
    else:
        newImage[where(originalImage + amountToBrighten > 0)] += float64(amountToBrighten)
        newImage[where(originalImage + amountToBrighten <= 0)] = 0.
    return uint8(newImage)


def histogramEqualization(originalImage, space):
    from cv2 import equalizeHist
    equalized = equalizeHist(originalImage)
    return equalized


def gaussianBlur(originalImage, space):
    from cv2 import GaussianBlur
    from random import randint
    min_blur = int(space['blur_extent_min'])
    max_blur = min_blur + int(space['blur_extent_distance'])
    amountToBlur = randint(min_blur, max_blur)
    if amountToBlur % 2 == 0: amountToBlur += 1
    return GaussianBlur(originalImage, (int(amountToBlur), int(amountToBlur)),0)


def horizontalFlip(originalImage, space):
    from cv2 import flip
    return flip(originalImage, flipCode=1)


def applyRandomAugmentation(originalImage, space):
    from random import uniform

    should_run = lambda pctprob: (uniform(0, 99)) <= pctprob

    augmentations = [gaussianBlur, histogramEqualization, addConstantOcclusion, changeBrightness, randomNoise, horizontalFlip]
    for fn in augmentations:
        if should_run(space[fn.__name__ + '_pctprob']):
            originalImage = fn(originalImage, space)

    return originalImage
