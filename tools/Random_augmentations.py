import random
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw, PIL.ImageFilter, PIL.ImageGrab
import numpy as np
import torch
from PIL import Image

"""
Reference : https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
            https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py
"""


def blend(image1, image2, factor):
    """Blend image1 and image2 using 'factor'.

    Factor can be above 0.0.  A value of 0.0 means only image1 is used.
    A value of 1.0 means only image2 is used.  A value between 0.0 and
    1.0 means we linearly interpolate the pixel values between the two
    images.  A value greater than 1.0 "extrapolates" the difference
    between the two pixel values, and we clip the results to values
    between 0 and 255.

    Args:
      image1: An image Tensor of type uint8.
      image2: An image Tensor of type uint8.
      factor: A floating point value above 0.0.

    Returns:
      A blended image Tensor of type uint8.
    """
    if factor == 0.0:
        return image1
    if factor == 1.0:
        return image2

    image1 = image1.float()
    image2 = image2.float()

    difference = image2 - image1
    scaled = factor * difference

    # Do addition in float.
    temp = image1 + scaled

    # Interpolate
    if 0.0 < factor < 1.0:
        # Interpolation means we always stay within 0 and 255.
        return temp.type(torch.uint8)

    # Extrapolate:
    #
    # We need to clip and then cast.
    temp = torch.clamp(temp, 0.0, 255.0)
    return temp.type(torch.uint8)


def random_noise(image, v):
    assert 0.0 <= v <= 0.4
    factor = v
    image = np.array(image)
    size = image.shape
    rand = np.random.uniform(-50, 50, size)
    rand1 = np.array([rand[:, :, 0], rand[:, :, 0], rand[:, :, 0]]).transpose((1, 2, 0))
    # print(rand[:,:,0])
    # print(rand)
    # rand = np.random.normal(0, 50, size)
    image1 = image + rand1 * factor
    image1 = Image.fromarray(np.uint8(image1))
    return image1


def gaussian_noise(image, v):
    assert 0.0 <= v <= 0.4
    factor = v
    image = np.array(image)
    size = image.shape
    # rand = np.random.uniform(-50, 50, size)
    rand = np.random.normal(0, 50, size)
    image1 = image + rand * factor
    image1 = Image.fromarray(np.uint8(image1))
    return image1


def gaussian_blur(image, v):
    assert 0.0 <= v <= 2.0
    image1 = image.filter(PIL.ImageFilter.GaussianBlur(radius=v))
    return image1


def scale(image, v):  # [0.9, 1.4]
    height, width = image.size[1], image.size[0]
    # print(height, width)
    new_height = round(height * v)
    new_width = round(width * v)
    if new_width < 224:
        new_width = 224
    if new_height < 224:
        new_height = 224
    image1 = image.resize((new_width, new_height), Image.ANTIALIAS)
    return image1


def scale_xy_diff(image, v):# [0.9, 1.2]
    factor_y = random.uniform(0.9, 1.4)
    factor_x = random.uniform(0.9, 1.4)
    height, width = image.size[1], image.size[0]
    # print(height,width)
    new_height = round(height * factor_x)
    new_width = round(width * factor_y)
    if new_width < 224:
        new_width = 224
    if new_height < 224:
        new_height = 224
    image1 = image.resize((new_width, new_height), Image.ANTIALIAS)
    return image1


def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def H_Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)


def V_Flip(img, _):
    return PIL.ImageOps.flip(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def SolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def Identity(img, v):
    return img


def original_augment_list():  # 16 oeprations and their ranges
    l = [
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Invert, 0, 1),
        (Rotate, 0, 30),
        (Posterize, 0, 4),
        (Solarize, 0, 256),
        (SolarizeAdd, 0, 110),
        (Contrast, 0.1, 1.9),
        (Color, 0.1, 1.9),
        (Brightness, 0.1, 1.9),
        (Sharpness, 0.1, 1.9),
        (ShearX, 0., 0.3),
        (ShearY, 0., 0.3),
        (CutoutAbs, 0, 40),
        (TranslateXabs, 0., 100),
        (TranslateYabs, 0., 100),
    ]

    return l


def augment_list_23():  # 23 oeprations and their ranges
    l = [
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Invert, 0, 1),
        (Rotate, 0, 30),
        (Posterize, 0, 4),
        (Solarize, 0, 256),
        (SolarizeAdd, 0, 110),
        (Contrast, 0.1, 1.9),
        (Color, 0.1, 1.9),  # 11
        (Brightness, 0.1, 1.9),
        (Sharpness, 0.1, 1.9),
        (ShearX, 0., 0.3),
        (ShearY, 0., 0.3),
        (CutoutAbs, 0, 40),
        (TranslateXabs, 0., 100),
        (TranslateYabs, 0., 100),
        # new
        (scale, 0.9, 1.4),
        (scale_xy_diff, 0.9, 1.4),
        (H_Flip, 0, 1),
        (V_Flip, 0, 1),
        (random_noise, 0., 0.4),
        (gaussian_noise, 0., 0.4),
        (gaussian_blur, 0., 2.0)
    ]

    return l


def color_augment_list(): # 13
    l = [
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Invert, 0, 1),
        (Posterize, 0, 4),
        (Solarize, 0, 256),
        (SolarizeAdd, 0, 110),
        (Color, 0.1, 1.9),
        (Contrast, 0.1, 1.9),
        (Brightness, 0.1, 1.9),
        (Sharpness, 0.1, 1.9),
        (random_noise, 0., 0.4),
        (gaussian_noise, 0., 0.4),
        (gaussian_blur, 0., 2.0)
    ]
    return l


def shape_augment_list(): # 10
    l = [
        (H_Flip, 0, 1),
        (V_Flip, 0, 1),
        (Rotate, 0, 30),
        (ShearX, 0., 0.3),
        (ShearY, 0., 0.3),
        (CutoutAbs, 0, 40),
        (TranslateXabs, 0., 100),
        (TranslateYabs, 0., 100),
        (scale, 0.9, 1.4),
        (scale_xy_diff, 0.9, 1.4)
    ]
    return l


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class RandAugment:
    '''
    n: Integer, the number of augmentation transformations to apply
      sequentially to an image. Represented as (N) in the paper.
    m: Integer, shared magnitude across all augmentation operations.
      Represented as (M) in the paper. Usually best values are in the range
      [5, 30].
    '''
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list = original_augment_list()
        print('kinds of transformations  is ', len(original_augment_list()))

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)

        return img


class RandAugment_23:
    '''
    n: Integer, the number of augmentation transformations to apply
      sequentially to an image. Represented as (N) in the paper.
    m: Integer, shared magnitude across all augmentation operations.
      Represented as (M) in the paper. Usually best values are in the range
      [5, 30].
    '''
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list = augment_list_23()
        print('kinds of transformations  is', len(augment_list_23()))

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            # print(val)
            img = op(img, val)

        return img


class Random_RandAugment:
    '''
    n: Integer, the number of augmentation transformations to apply
      sequentially to an image. Represented as (N) in the paper.
    m: Integer, shared magnitude across all augmentation operations.
      Represented as (M) in the paper. Usually best values are in the range
      [5, 30].
    '''
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.augment_list = augment_list_23()
        print('kinds of transformations  is ', len(augment_list_23()))

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        self.m = random.randint(5, self.m)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            # print(val)
            img = op(img, val)

        return img


class Two_stage_Random_RandAugment:

    '''
    n: Integer, the number of augmentation transformations to apply
      sequentially to an image. Represented as (N) in the paper.
    m: Integer, shared magnitude across all augmentation operations.
      Represented as (M) in the paper. Usually best values are in the range
      [5, 30].
    '''

    def __init__(self, p, n_color, n_shape):
        self.p = p
        self.color_augment_list = color_augment_list()
        self.shape_augment_list = shape_augment_list()
        self.n_color = n_color
        self.n_shape = n_shape
        print('color and shape transforms are ', len(color_augment_list()), len(shape_augment_list()))

    def __call__(self, img):
        ops1 = random.choices(self.color_augment_list, k=self.n_color)
        self.m = random.randint(5, 30)
        for op, minval, maxval in ops1:
            should_apply_op1 = random.random() + self.p
            if should_apply_op1 >= 1.0:
                val = (float(self.m) / 30) * float(maxval - minval) + minval
                # print(op)
                img = op(img, val)
        ops2 = random.choices(self.shape_augment_list, k=self.n_shape)
        self.m = random.randint(5, 30)
        for op, minval, maxval in ops2:
            should_apply_op2 = random.random() + self.p
            if should_apply_op2 >= 1.0:
                val = (float(self.m) / 30) * float(maxval - minval) + minval
                # print(op)
                img = op(img, val)

        return img
