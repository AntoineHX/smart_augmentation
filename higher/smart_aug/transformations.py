""" PyTorch implementation of some PIL image transformations.

    Those implementation are thinked to take advantages of batched computation of PyTorch on GPU.

    Based on Kornia library.
    See: https://github.com/kornia/kornia

    And PIL.
    See: 
        https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py
        https://github.com/python-pillow/Pillow/blob/9c78c3f97291bd681bc8637922d6a2fa9415916c/src/PIL/Image.py#L2818

    Inspired from AutoAugment.
    See: https://github.com/tensorflow/models/blob/fc2056bce6ab17eabdc139061fef8f4f2ee763ec/research/autoaugment/augmentation_transforms.py
"""

import torch
import kornia
import random
import json

#TF that don't have use for magnitude parameter.
TF_no_mag={'Identity', 'FlipUD', 'FlipLR', 'Random', 'RandBlend', 'identity', 'flip'}
#TF which implemetation doesn't allow gradient propagaition. 
TF_no_grad={'Solarize', 'Posterize', '=Solarize', '=Posterize', 'posterize','solarize'} #Numpy implementation would be better ?
#TF for which magnitude should be ignored (Magnitude fixed).
TF_ignore_mag= TF_no_mag | TF_no_grad 

# What is the max 'level' a transform could be predicted
PARAMETER_MAX = 1
# What is the min 'level' a transform could be predicted
PARAMETER_MIN = 0.01 

#Dict containing the value for wich TF are closer to identity
#TF_identity={
#    PARAMETER_MAX:{'Solarize', 'Posterize'},
#    PARAMETER_MAX/2:{'Contrast','Color','Brightness','Sharpness'},
#    PARAMETER_MIN:{'Rotate','TranslateX','TranslateY','ShearX','ShearY'},
#}

class Normalizer(object):
    def __init__(self, mean, std):
        self.mean=torch.tensor(mean)
        self.std=torch.tensor(std)
    def __call__(self, x):
        # return x.sub_(self.mean.to(x.device)[None, :, None, None]).div_(self.std.to(x.device)[None, :, None, None])
        return kornia.color.normalize(x, self.mean, self.std)
    def reverse(self, x):
        # return x.mul_(self.std.to(x.device)[None, :, None, None]).add_(self.mean.to(x.device)[None, :, None, None])
        return kornia.color.denormalize(x, self.mean, self.std).to(torch.half).to(torch.float)

class TF_loader(object):
    """ Transformations builder.

        See 'config' folder for pre-defined config files.

        Attributes:
            _filename (str): Path to config file (JSON) used.
            _TF_dict (dict): Transformations dictionnary built from config file.
            _TF_ignore_mag (set): Ensemble of transformations names for which magnitude should be ignored.
            _TF_names (list): List of transformations names/keys.
    """
    def __init__(self):
        """ Initialize TF_loader.

        """
        self._filename=''
        self._TF_dict={}
        self._TF_ignore_mag=set()
        self._TF_names=[]

    def load_TF_dict(self, filename):
        """ Build a TF dictionnary.

            Args:
                filename (str): Path to config file (JSON) defining the transformations.
            Returns:
                (dict, set) : TF dicttionnary built and ensemble of TF names for which mag should be ignored.
        """
        self._filename=filename
        self._TF_names=[]
        self._TF_dict={}
        self._TF_ignore_mag=set()

        with open(filename) as json_file:
            TF_params = json.load(json_file)

        for tf in TF_params:
            self._TF_names.append(tf['name'])
            if tf['function'] in TF_ignore_mag:
                self._TF_ignore_mag.add(tf['name'])

            if tf['function'] == 'identity':
                self._TF_dict[tf['name']]=(lambda x, mag: x)

            elif tf['function'] == 'flip':
                #Inverser axes ?
                if tf['param']['axis'] == 'X':
                    self._TF_dict[tf['name']]=(lambda x, mag: flipLR(x))
                elif tf['param']['axis'] == 'Y':
                    self._TF_dict[tf['name']]=(lambda x, mag: flipUD(x))
                else:
                    raise Exception("Unknown TF axis : %s in %s"%(tf['function'], self._filename))

            # elif tf['function'] in {'translate', 'shear'}:
            #     rand_fct= 'invScale_rand_floats' if tf['param']['invScale'] else 'rand_floats'
            #     self._TF_dict[tf['name']]=self.build_lambda(tf['function'], rand_fct, tf['param']['min'], tf['param']['max'], tf['param']['absolute'], tf['param']['axis'])
            
            else:
                axis = tf['param']['axis'] if 'axis' in tf['param'].keys() else None
                absolute = tf['param']['absolute'] if 'absolute' in tf['param'].keys() else True
                rand_fct= 'invScale_rand_floats' if tf['param']['invScale'] else 'rand_floats'
                self._TF_dict[tf['name']]=self.build_lambda(tf['function'], rand_fct, tf['param']['min'], tf['param']['max'], absolute, axis)

        return self._TF_dict, self._TF_ignore_mag

    def build_lambda(self, fct_name, rand_fct_name, minval, maxval, absolute=True, axis=None):
        """ Build a lambda function performing transformations.

            Force different context for creation of each lambda function.

            Args:
                fct_name (str): Name of the transformations to use (see transformations.py).
                rand_fct_name (str): Name of the random mapping function to use (see transformations.py).
                minval (float): minimum magnitude value of the TF.
                maxval (float): maximum magnitude value of the TF.
                absolute (bool): Wether the maxval should be relative (absolute=False) to the image size. (default: True)
                axis (str): Axis ('X' / 'Y') of the TF, if relevant. Should be used for (flip)/translate/shear functions. (default: None)

            Returns:
                (function) transformations function : Tensor=f(Tensor, magnitude)
        """
        if absolute:
            max_val_fct=(lambda x: maxval)
        else: #Relative to img size
            max_val_fct=(lambda x: x*maxval)

        if axis is None:
            return (lambda x, mag: 
                        globals()[fct_name]( #getattr(TF, fct_name)
                            x, 
                            globals()[rand_fct_name](
                                size=x.shape[0], 
                                mag=mag, 
                                minval=minval, 
                                maxval=max_val_fct(max(x.shape[2],x.shape[3]))))) 
        elif axis =='X':
            return (lambda x, mag: 
                globals()[fct_name](
                    x, 
                    zero_stack(
                        globals()[rand_fct_name](
                            size=(x.shape[0],), 
                            mag=mag, 
                            minval=minval, 
                            maxval=max_val_fct(x.shape[2])),
                        zero_pos=0)))
        elif axis == 'Y':
            return (lambda x, mag: 
                globals()[fct_name](
                    x, 
                    zero_stack(
                        globals()[rand_fct_name](
                            size=(x.shape[0],), 
                            mag=mag, 
                            minval=minval, 
                            maxval=max_val_fct(x.shape[3])),
                        zero_pos=1)))
        else:
            raise Exception("Unknown TF axis : %s in %s"%(fct_name, self._filename))

        def get_TF_names(self):
            return self._TF_names
        def get_TF_dict(self):
            return self._TF_dict
## Image type cast ##
def int_image(float_image):
    """Convert a float Tensor/Image to an int Tensor/Image.

    Be warry that this transformation isn't bijective, each conversion will result in small loss of information.
    Granularity: 1/256 = 0.0039.

    This will also result in the loss of the gradient associated to input as gradient cannot be tracked on int Tensor.

    Args:
        float_image (FloatTensor): Image tensor.

    Returns:
        (ByteTensor) Converted tensor.
    """
    return (float_image*255.).type(torch.uint8)

def float_image(int_image):
    """Convert a int Tensor/Image to an float Tensor/Image.

        Args:
            int_image (ByteTensor): Image tensor.

        Returns:
            (FloatTensor) Converted tensor.
    """
    return int_image.type(torch.float)/255.

## Parameters utils ##
def rand_floats(size, mag, maxval, minval=None):
    """Generate a batch of random values.

        Args:
            size (int): Number of value to generate.
            mag (float): Level of the operation that will be between [PARAMETER_MIN, PARAMETER_MAX].
            maxval (float): Maximum value that can be generated. This will be scaled to mag/PARAMETER_MAX.
            minval (float): Minimum value that can be generated. (default: -maxval)

        Returns:
            (Tensor) Generated batch of float values between [minval, maxval].
    """
    real_mag = float_parameter(mag, maxval=maxval)
    if not minval : minval = -real_mag
    #return random.uniform(minval, real_max)
    return minval + (real_mag-minval) * torch.rand(size, device=mag.device) #[min_val, real_mag]

def invScale_rand_floats(size, mag, maxval, minval):
    """Generate a batch of random values.

        Similar to rand_floats() except that the mag is used in an inversed scale.

        Mag:[0,PARAMETER_MAX] => [PARAMETER_MAX, 0]

        Args:
            size (int): Number of value to generate.
            mag (float): Level of the operation that will be between [PARAMETER_MIN, PARAMETER_MAX].
            maxval (float): Maximum value that can be generated. This will be scaled to mag/PARAMETER_MAX.
            minval (float): Minimum value that can be generated. (default: -maxval)

        Returns:
            (Tensor) Generated batch of float values between [minval, maxval].
    """
    real_mag = float_parameter(float(PARAMETER_MAX) - mag, maxval=maxval-minval)+minval 
    return real_mag + (maxval-real_mag) * torch.rand(size, device=mag.device) #[real_mag, max_val]

def zero_stack(tensor, zero_pos):
    """Add a row of zeros to a Tensor.

        This function is intended to be used with single row Tensor, thus returning a 2 dimension Tensor.

        Args:
            tensor (Tensor): Tensor to be stacked with zeros.
            zero_pos (int): Wheter the zeros should be added before or after the Tensor. Either 0 or 1.

        Returns:
            Stacked Tensor.
    """
    if zero_pos==0:
        return torch.stack((tensor, torch.zeros((tensor.shape[0],), device=tensor.device)), dim=1)
    if zero_pos==1:
        return torch.stack((torch.zeros((tensor.shape[0],), device=tensor.device), tensor), dim=1)
    else:
        raise Exception("Invalid zero_pos : ", zero_pos) 
    
def float_parameter(level, maxval):
    """Scale level between 0 and maxval.

        Args:
            level (float): Level of the operation that will be between [PARAMETER_MIN, PARAMETER_MAX].
            maxval: Maximum value that the operation can have. This will be scaled to level/PARAMETER_MAX.
        Returns:
            A float that results from scaling `maxval` according to `level`.
    """

    #return float(level) * maxval / PARAMETER_MAX
    return (level * maxval / PARAMETER_MAX)#.to(torch.float) 

## Tranformations ##
def flipLR(x):
    """Flip horizontaly/Left-Right images.
        
        Args:
            x (Tensor): Batch of images.

        Returns: 
            (Tensor): Batch of fliped images.
    """
    device = x.device
    (batch_size, channels, h, w) = x.shape

    M =torch.tensor( [[[-1.,  0., w-1],
                        [ 0.,  1.,  0.],
                        [ 0.,  0.,  1.]]], device=device).expand(batch_size,-1,-1)

    # warp the original image by the found transform
    return kornia.warp_perspective(x, M, dsize=(h, w))

def flipUD(x):
    """Flip vertically/Up-Down images.
        
        Args:
            x (Tensor): Batch of images.

        Returns: 
            (Tensor): Batch of fliped images.
    """
    device = x.device
    (batch_size, channels, h, w) = x.shape

    M =torch.tensor( [[[ 1.,  0.,  0.],
                        [ 0., -1.,  h-1],
                        [ 0.,  0.,  1.]]], device=device).expand(batch_size,-1,-1)

    # warp the original image by the found transform
    return kornia.warp_perspective(x, M, dsize=(h, w))

def rotate(x, angle):
    """Rotate images.

        Args:
            x (Tensor): Batch of images.
            angle (Tensor): Angles (degrees) of rotation for each images.

        Returns:
            (Tensor): Batch of rotated images.
    """
    return kornia.rotate(x, angle=angle.type(torch.float)) #Kornia ne supporte pas les int

def translate(x, translation):
    """Translate images.

        Args:
            x (Tensor): Batch of images.
            translation (Tensor): Distance (pixels) of translation for each images.

        Returns:
            (Tensor): Batch of translated images.
    """
    return kornia.translate(x, translation=translation.type(torch.float)) #Kornia ne supporte pas les int

def shear(x, shear):
    """Shear images.

    Args:
        x (Tensor): Batch of images.
        shear (Tensor): Angle of shear for each images.

    Returns:
        (Tensor): Batch of skewed images.
    """
    return kornia.shear(x, shear=shear)

def contrast(x, contrast_factor):
    """Adjust contast of images.

    Args:
        x (FloatTensor): Batch of images.
        contrast_factor (FloatTensor): Contrast adjust factor per element in the batch. 
        0 generates a compleatly black image, 1 does not modify the input image while any other non-negative number modify the brightness by this factor.

    Returns:
        (Tensor): Batch of adjusted images.
    """
    return kornia.adjust_contrast(x, contrast_factor=contrast_factor) #Expect image in the range of [0, 1]

def color(x, color_factor):
    """Adjust color of images.

    Args:
        x (Tensor): Batch of images.
        color_factor (Tensor): Color factor for each images. 
        0.0 gives a black and white image. A factor of 1.0 gives the original image.

    Returns:
        (Tensor): Batch of adjusted images.
    """
    (batch_size, channels, h, w) = x.shape

    gray_x = kornia.rgb_to_grayscale(x)
    gray_x = gray_x.repeat_interleave(channels, dim=1)
    return blend(gray_x, x, color_factor).clamp(min=0.0,max=1.0) #Expect image in the range of [0, 1]

def brightness(x, brightness_factor):
    """Adjust brightness of images.

    Args:
        x (Tensor): Batch of images.
        brightness_factor (Tensor): Brightness factor for each images. 
        0.0 gives a black image. A factor of 1.0 gives the original image.

    Returns:
        (Tensor): Batch of adjusted images.
    """
    device = x.device

    return blend(torch.zeros(x.size(), device=device), x, brightness_factor).clamp(min=0.0,max=1.0) #Expect image in the range of [0, 1]

def sharpness(x, sharpness_factor):
    """Adjust sharpness of images.

    Args:
        x (Tensor): Batch of images.
        sharpness_factor (Tensor): Sharpness factor for each images. 
        0.0 gives a black image. A factor of 1.0 gives the original image.

    Returns:
        (Tensor): Batch of adjusted images.
    """
    device = x.device
    (batch_size, channels, h, w) = x.shape

    k = torch.tensor([[[ 1.,  1.,  1.],
                       [ 1.,  5.,  1.],
                       [ 1.,  1.,  1.]]], device=device) #Smooth Filter : https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageFilter.py
    smooth_x = kornia.filter2D(x, kernel=k, border_type='reflect', normalized=True) #Peut etre necessaire de s'occuper du channel Alhpa differement

    return blend(smooth_x, x, sharpness_factor).clamp(min=0.0,max=1.0) #Expect image in the range of [0, 1]

def posterize(x, bits):
    """Reduce the number of bits for each color channel.

        Be warry that the cast to integers block the gradient propagation.
    Args:
        x (Tensor): Batch of images.
        bits (Tensor): The number of bits to keep for each channel (1-8).

    Returns:
        (Tensor): Batch of posterized images.
    """
    bits = bits.type(torch.uint8) #Perte du gradient
    x = int_image(x) #Expect image in the range of [0, 1]

    mask = ~(2 ** (8 - bits) - 1).type(torch.uint8)

    (batch_size, channels, h, w) = x.shape
    mask = mask.unsqueeze(dim=1).expand(-1,channels).unsqueeze(dim=2).expand(-1,channels, h).unsqueeze(dim=3).expand(-1,channels, h, w) #Il y a forcement plus simple ...

    return float_image(x & mask)

def cutout(img, length):
    """
    Args:
        img (Tensor): Batch of images. Expect image value between [0, 1].
        length (Tensor): The length (in pixels) of each square patch.
    Returns:
        Tensor: Images with single holes of dimension length x length cut out of it.
    """
    device = img.device
    (batch_size, channels, h, w) = img.shape

    # mask = np.ones((h, w), np.float32)
    mask = torch.ones((batch_size, h, w), device=device)
    length=length.type(torch.uint8)

    # y = np.random.randint(h)
    # x = np.random.randint(w)
    y = torch.randint(low=0, high=h, size=(batch_size,)).to(device)
    x = torch.randint(low=0, high=w, size=(batch_size,)).to(device)

    # y1 = np.clip(y - length // 2, 0, h)
    # y2 = np.clip(y + length // 2, 0, h)
    # x1 = np.clip(x - length // 2, 0, w)
    # x2 = np.clip(x + length // 2, 0, w)
    y1 = (y - length // 2).clamp(min=0, max=h)
    y2 = (y + length // 2).clamp(min=0, max=h)
    x1 = (x - length // 2).clamp(min=0, max=w)
    x2 = (x + length // 2).clamp(min=0, max=w)

    # mask[y1: y2, x1: x2] = 0.
    for idx in range(batch_size): #Pas opti pour des batch
        mask[idx, y1[idx]: y2[idx], x1[idx]: x2[idx]]= 0.

    # mask = torch.from_numpy(mask)
    # mask = mask.expand_as(img)
    mask = mask.unsqueeze(dim=1).expand_as(img)
    img = img * mask

    return img

import torch.nn.functional as F
def solarize(x, thresholds):
    """Invert all pixel values above a threshold.

        Be warry that the use of the inequality (x>tresholds) block the gradient propagation.
        
        TODO : Make differentiable.

    Args:
        x (Tensor): Batch of images.
        thresholds (Tensor):  All pixels above this level are inverted

    Returns:
        (Tensor): Batch of solarized images.
    """
    batch_size, channels, h, w = x.shape
    #imgs=[]
    #for idx, t in enumerate(thresholds): #Operation par image
    #  mask = x[idx] > t #Perte du gradient
    #In place
    #  inv_x = 1-x[idx][mask]
    #  x[idx][mask]=inv_x
    #

    #Out of place
    #  im = x[idx]
    #  inv_x = 1-im[mask]

    #  imgs.append(im.masked_scatter(mask,inv_x))

    #idxs=torch.tensor(range(x.shape[0]), device=x.device)
    #idxs=idxs.unsqueeze(dim=1).expand(-1,channels).unsqueeze(dim=2).expand(-1,channels, h).unsqueeze(dim=3).expand(-1,channels, h, w) #Il y a forcement plus simple ...
    #x=x.scatter(dim=0, index=idxs, src=torch.stack(imgs))
    #

    thresholds = thresholds.unsqueeze(dim=1).expand(-1,channels).unsqueeze(dim=2).expand(-1,channels, h).unsqueeze(dim=3).expand(-1,channels, h, w) #Il y a forcement plus simple ...
    x=torch.where(x>thresholds,1-x, x)

    #x=x.min(thresholds)
    #inv_x = 1-x[mask]
    #x=x.where(x<thresholds,1-x)
    #x[mask]=inv_x
    #x=x.masked_scatter(mask, inv_x)

    #Differentiable (/Thresholds) ?

    #inv_x_bT= F.relu(x) - F.relu(x - thresholds)
    #inv_x_aT= 1-x #Besoin thresholds

    #print('-'*10)
    #print(thresholds[0])
    #print(x[0])
    #print(inv_x_bT[0])
    #print(inv_x_aT[0])

    #x=torch.where(x>thresholds,inv_x_aT, inv_x_bT)
    #print(torch.allclose(x, x+0.001, atol=1e-3))
    #print(torch.allclose(x, sol_x, atol=1e-2))
    #print(torch.eq(x,sol_x)[0])

    #print(x[0])
    #print(sol_x[0])
    #'''
    return x

def blend(x,y,alpha):
    """Creates a new images by interpolating between two input images, using a constant alpha.
        
        x and y should have the same size.
        alpha should have the same batch size as the images.

        Apply batch wise :
            out = image1 * (1.0 - alpha) + image2 * alpha

    Args:
        x (Tensor): Batch of images.
        y (Tensor): Batch of images.
        alpha (Tensor):  The interpolation alpha factor for each images.
    Returns:
        (Tensor): Batch of solarized images.
    """
    #return kornia.add_weighted(src1=x, alpha=(1-alpha), src2=y, beta=alpha, gamma=0) #out=src1∗alpha+src2∗beta+gamma #Ne fonctionne pas pour des batch de alpha

    if not isinstance(x, torch.Tensor):
        raise TypeError("x should be a tensor. Got {}".format(type(x)))

    if not isinstance(y, torch.Tensor):
        raise TypeError("y should be a tensor. Got {}".format(type(y)))

    assert(x.shape==y.shape and x.shape[0]==alpha.shape[0])

    (batch_size, channels, h, w) = x.shape
    alpha = alpha.unsqueeze(dim=1).expand(-1,channels).unsqueeze(dim=2).expand(-1,channels, h).unsqueeze(dim=3).expand(-1,channels, h, w) #Il y a forcement plus simple ...
    res = x*(1-alpha) + y*alpha

    return res

#Not working
def auto_contrast(x):
    """NOT TESTED - EXTRA SLOW

    """
    # Optimisation : Application de LUT efficace / Calcul d'histogramme par batch/channel
    print("Warning : Pas encore check !")
    (batch_size, channels, h, w) = x.shape
    x = int_image(x) #Expect image in the range of [0, 1]
    #print('Start',x[0])
    for im_idx, img in enumerate(x.chunk(batch_size, dim=0)): #Operation par image
        #print(img.shape)
        for chan_idx, chan in enumerate(img.chunk(channels, dim=1)): # Operation par channel
            #print(chan.shape)
            hist = torch.histc(chan, bins=256, min=0, max=255) #PAS DIFFERENTIABLE

            # find lowest/highest samples after preprocessing
            for lo in range(256):
                if hist[lo]:
                    break
            for hi in range(255, -1, -1):
                if hist[hi]:
                    break
                if hi <= lo:
                    # don't bother
                    pass
            else:
                scale = 255.0 / (hi - lo)
                offset = -lo * scale
                for ix in range(256):
                    n_ix = int(ix * scale + offset)
                    if n_ix < 0: n_ix = 0
                    elif n_ix > 255: n_ix = 255

            chan[chan==ix]=n_ix
            x[im_idx, chan_idx]=chan

    #print('End',x[0])
    return float_image(x)

def equalize(x):
    """ NOT WORKING

    """
    raise Exception(self, "not implemented") 
    # Optimisation : Application de LUT efficace / Calcul d'histogramme par batch/channel
    (batch_size, channels, h, w) = x.shape
    x = int_image(x) #Expect image in the range of [0, 1]
    #print('Start',x[0])
    for im_idx, img in enumerate(x.chunk(batch_size, dim=0)): #Operation par image
        #print(img.shape)
        for chan_idx, chan in enumerate(img.chunk(channels, dim=1)): # Operation par channel
            #print(chan.shape)
            hist = torch.histc(chan, bins=256, min=0, max=255) #PAS DIFFERENTIABLE

    return float_image(x)

'''
# Dictionnary mapping tranformations identifiers to their function.
# Each value of the dict should be a lambda function taking a (batch of data, magnitude of transformations) tuple as input and returns a batch of data.
TF_dict={ #Dataugv5+
    ## Geometric TF ##
    'Identity' : (lambda x, mag: x),
    'FlipUD' : (lambda x, mag: flipUD(x)),
    'FlipLR' : (lambda x, mag: flipLR(x)),
    'Rotate': (lambda x, mag: rotate(x, angle=rand_floats(size=x.shape[0], mag=mag, maxval=30))),
    'TranslateX': (lambda x, mag: translate(x, translation=zero_stack(rand_floats(size=(x.shape[0],), mag=mag, maxval=x.shape[2]*0.33), zero_pos=0))),
    'TranslateY': (lambda x, mag: translate(x, translation=zero_stack(rand_floats(size=(x.shape[0],), mag=mag, maxval=x.shape[3]*0.33), zero_pos=1))),
    'TranslateXabs': (lambda x, mag: translate(x, translation=zero_stack(rand_floats(size=(x.shape[0],), mag=mag, maxval=20), zero_pos=0))),
    'TranslateYabs': (lambda x, mag: translate(x, translation=zero_stack(rand_floats(size=(x.shape[0],), mag=mag, maxval=20), zero_pos=1))),
    'ShearX': (lambda x, mag: shear(x, shear=zero_stack(rand_floats(size=(x.shape[0],), mag=mag, maxval=0.3), zero_pos=0))),
    'ShearY': (lambda x, mag: shear(x, shear=zero_stack(rand_floats(size=(x.shape[0],), mag=mag, maxval=0.3), zero_pos=1))),

    ## Color TF (Expect image in the range of [0, 1]) ##
    'Contrast': (lambda x, mag: contrast(x, contrast_factor=rand_floats(size=x.shape[0], mag=mag, minval=0.1, maxval=1.9))),
    'Color':(lambda x, mag: color(x, color_factor=rand_floats(size=x.shape[0], mag=mag, minval=0.1, maxval=1.9))),
    'Brightness':(lambda x, mag: brightness(x, brightness_factor=rand_floats(size=x.shape[0], mag=mag, minval=0.1, maxval=1.9))),
    'Sharpness':(lambda x, mag: sharpness(x, sharpness_factor=rand_floats(size=x.shape[0], mag=mag, minval=0.1, maxval=1.9))),
    'Posterize': (lambda x, mag: posterize(x, bits=rand_floats(size=x.shape[0], mag=mag, minval=4., maxval=8.))),#Perte du gradient
    'Solarize': (lambda x, mag: solarize(x, thresholds=rand_floats(size=x.shape[0], mag=mag, minval=1/256., maxval=256/256.))), #Perte du gradient #=>Image entre [0,1]

    #Color TF (Common mag scale)
    '+Contrast': (lambda x, mag: contrast(x, contrast_factor=rand_floats(size=x.shape[0], mag=mag, minval=1.0, maxval=1.9))),
    '+Color':(lambda x, mag: color(x, color_factor=rand_floats(size=x.shape[0], mag=mag, minval=1.0, maxval=1.9))),
    '+Brightness':(lambda x, mag: brightness(x, brightness_factor=rand_floats(size=x.shape[0], mag=mag, minval=1.0, maxval=1.9))),
    '+Sharpness':(lambda x, mag: sharpness(x, sharpness_factor=rand_floats(size=x.shape[0], mag=mag, minval=1.0, maxval=1.9))),
    '-Contrast': (lambda x, mag: contrast(x, contrast_factor=invScale_rand_floats(size=x.shape[0], mag=mag, minval=0.1, maxval=1.0))),
    '-Color':(lambda x, mag: color(x, color_factor=invScale_rand_floats(size=x.shape[0], mag=mag, minval=0.1, maxval=1.0))),
    '-Brightness':(lambda x, mag: brightness(x, brightness_factor=invScale_rand_floats(size=x.shape[0], mag=mag, minval=0.1, maxval=1.0))),
    '-Sharpness':(lambda x, mag: sharpness(x, sharpness_factor=invScale_rand_floats(size=x.shape[0], mag=mag, minval=0.1, maxval=1.0))),
    '=Posterize': (lambda x, mag: posterize(x, bits=invScale_rand_floats(size=x.shape[0], mag=mag, minval=4., maxval=8.))),#Perte du gradient
    '=Solarize': (lambda x, mag: solarize(x, thresholds=invScale_rand_floats(size=x.shape[0], mag=mag, minval=1/256., maxval=256/256.))), #Perte du gradient #=>Image entre [0,1]

    ## Bad Tranformations ##
    # Bad Geometric TF #
    'BShearX': (lambda x, mag: shear(x, shear=zero_stack(rand_floats(size=(x.shape[0],), mag=mag, minval=0.3*3, maxval=0.3*4), zero_pos=0))),
    'BShearY': (lambda x, mag: shear(x, shear=zero_stack(rand_floats(size=(x.shape[0],), mag=mag, minval=0.3*3, maxval=0.3*4), zero_pos=1))),
    'BTranslateX': (lambda x, mag: translate(x, translation=zero_stack(rand_floats(size=(x.shape[0],), mag=mag, minval=25, maxval=30), zero_pos=0))),
    'BTranslateX-': (lambda x, mag: translate(x, translation=zero_stack(rand_floats(size=(x.shape[0],), mag=mag, minval=-25, maxval=-30), zero_pos=0))),
    'BTranslateY': (lambda x, mag: translate(x, translation=zero_stack(rand_floats(size=(x.shape[0],), mag=mag, minval=25, maxval=30), zero_pos=1))),
    'BTranslateY-': (lambda x, mag: translate(x, translation=zero_stack(rand_floats(size=(x.shape[0],), mag=mag, minval=-25, maxval=-30), zero_pos=1))),

    # Bad Color TF #
    'BadContrast': (lambda x, mag: contrast(x, contrast_factor=rand_floats(size=x.shape[0], mag=mag, minval=1.9*2, maxval=2*4))),
    'BadBrightness':(lambda x, mag: brightness(x, brightness_factor=rand_floats(size=x.shape[0], mag=mag, minval=1.9, maxval=2*3))),

    # Random TF #
    'Random':(lambda x, mag: torch.rand_like(x)),
    'RandBlend': (lambda x, mag: blend(x,torch.rand_like(x), alpha=torch.tensor(0.7,device=mag.device).expand(x.shape[0]))),

    #Not ready for use
    #'Auto_Contrast': (lambda mag: None), #Pas opti pour des batch (Super lent)
    #'Equalize': (lambda mag: None),
}
'''