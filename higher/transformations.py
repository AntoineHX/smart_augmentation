import torch
import kornia
import random

### Available TF for Dataug ###
'''
TF_dict={ #Dataugv4
  ## Geometric TF ##
  'Identity' : (lambda x, mag: x),
  'FlipUD' : (lambda x, mag: flipUD(x)),
  'FlipLR' : (lambda x, mag: flipLR(x)),
  'Rotate': (lambda x, mag: rotate(x, angle=torch.tensor([rand_int(mag, maxval=30)for _ in x], device=x.device))),
  'TranslateX': (lambda x, mag: translate(x, translation=torch.tensor([[rand_int(mag, maxval=20), 0] for _ in x], device=x.device))),
  'TranslateY': (lambda x, mag: translate(x, translation=torch.tensor([[0, rand_int(mag, maxval=20)] for _ in x], device=x.device))),
  'ShearX': (lambda x, mag: shear(x, shear=torch.tensor([[rand_float(mag, maxval=0.3), 0] for _ in x], device=x.device))),
  'ShearY': (lambda x, mag: shear(x, shear=torch.tensor([[0, rand_float(mag, maxval=0.3)] for _ in x], device=x.device))),

  ## Color TF (Expect image in the range of [0, 1]) ##
  'Contrast': (lambda x, mag: contrast(x, contrast_factor=torch.tensor([rand_float(mag, minval=0.1, maxval=1.9) for _ in x], device=x.device))),
  'Color':(lambda x, mag: color(x, color_factor=torch.tensor([rand_float(mag, minval=0.1, maxval=1.9) for _ in x], device=x.device))),
  'Brightness':(lambda x, mag: brightness(x, brightness_factor=torch.tensor([rand_float(mag, minval=0.1, maxval=1.9) for _ in x], device=x.device))),
  'Sharpness':(lambda x, mag: sharpeness(x, sharpness_factor=torch.tensor([rand_float(mag, minval=0.1, maxval=1.9) for _ in x], device=x.device))),
  'Posterize': (lambda x, mag: posterize(x, bits=torch.tensor([rand_int(mag, minval=4, maxval=8) for _ in x], device=x.device))),
  'Solarize': (lambda x, mag: solarize(x, thresholds=torch.tensor([rand_int(mag,minval=1, maxval=256)/256. for _ in x], device=x.device))) , #=>Image entre [0,1] #Pas opti pour des batch

  #Non fonctionnel
  #'Auto_Contrast': (lambda mag: None), #Pas opti pour des batch (Super lent)
  #'Equalize': (lambda mag: None),
}
'''
'''
TF_dict={ #Dataugv5 #AutoAugment
  ## Geometric TF ##
  'Identity' : (lambda x, mag: x),
  'FlipUD' : (lambda x, mag: flipUD(x)),
  'FlipLR' : (lambda x, mag: flipLR(x)),
  'Rotate': (lambda x, mag: rotate(x, angle=rand_floats(size=x.shape[0], mag=mag, maxval=30))),
  'TranslateX': (lambda x, mag: translate(x, translation=zero_stack(rand_floats(size=(x.shape[0],), mag=mag, maxval=20), zero_pos=0))),
  'TranslateY': (lambda x, mag: translate(x, translation=zero_stack(rand_floats(size=(x.shape[0],), mag=mag, maxval=20), zero_pos=1))),
  'ShearX': (lambda x, mag: shear(x, shear=zero_stack(rand_floats(size=(x.shape[0],), mag=mag, maxval=0.3), zero_pos=0))),
  'ShearY': (lambda x, mag: shear(x, shear=zero_stack(rand_floats(size=(x.shape[0],), mag=mag, maxval=0.3), zero_pos=1))),

  ## Color TF (Expect image in the range of [0, 1]) ##
  'Contrast': (lambda x, mag: contrast(x, contrast_factor=rand_floats(size=x.shape[0], mag=mag, minval=0.1, maxval=1.9))),
  'Color':(lambda x, mag: color(x, color_factor=rand_floats(size=x.shape[0], mag=mag, minval=0.1, maxval=1.9))),
  'Brightness':(lambda x, mag: brightness(x, brightness_factor=rand_floats(size=x.shape[0], mag=mag, minval=0.1, maxval=1.9))),
  'Sharpness':(lambda x, mag: sharpeness(x, sharpness_factor=rand_floats(size=x.shape[0], mag=mag, minval=0.1, maxval=1.9))),
  'Posterize': (lambda x, mag: posterize(x, bits=rand_floats(size=x.shape[0], mag=mag, minval=4., maxval=8.))),#Perte du gradient
  'Solarize': (lambda x, mag: solarize(x, thresholds=rand_floats(size=x.shape[0], mag=mag, minval=1/256., maxval=256/256.))), #Perte du gradient #=>Image entre [0,1] #Pas opti pour des batch
  
  #Non fonctionnel
  #'Auto_Contrast': (lambda mag: None), #Pas opti pour des batch (Super lent)
  #'Equalize': (lambda mag: None),
}
'''
TF_dict={ #Dataugv5
  ## Geometric TF ##
  'Identity' : (lambda x, mag: x),
  'FlipUD' : (lambda x, mag: flipUD(x)),
  'FlipLR' : (lambda x, mag: flipLR(x)),
  'Rotate': (lambda x, mag: rotate(x, angle=rand_floats(size=x.shape[0], mag=mag, maxval=30))),
  'TranslateX': (lambda x, mag: translate(x, translation=zero_stack(rand_floats(size=(x.shape[0],), mag=mag, maxval=20), zero_pos=0))),
  'TranslateY': (lambda x, mag: translate(x, translation=zero_stack(rand_floats(size=(x.shape[0],), mag=mag, maxval=20), zero_pos=1))),
  'ShearX': (lambda x, mag: shear(x, shear=zero_stack(rand_floats(size=(x.shape[0],), mag=mag, maxval=0.3), zero_pos=0))),
  'ShearY': (lambda x, mag: shear(x, shear=zero_stack(rand_floats(size=(x.shape[0],), mag=mag, maxval=0.3), zero_pos=1))),

  ## Color TF (Expect image in the range of [0, 1]) ##
  'Contrast': (lambda x, mag: contrast(x, contrast_factor=rand_floats(size=x.shape[0], mag=mag, minval=0.1, maxval=1.9))),
  'Color':(lambda x, mag: color(x, color_factor=rand_floats(size=x.shape[0], mag=mag, minval=0.1, maxval=1.9))),
  'Brightness':(lambda x, mag: brightness(x, brightness_factor=rand_floats(size=x.shape[0], mag=mag, minval=0.1, maxval=1.9))),
  'Sharpness':(lambda x, mag: sharpeness(x, sharpness_factor=rand_floats(size=x.shape[0], mag=mag, minval=0.1, maxval=1.9))),
  'Posterize': (lambda x, mag: posterize(x, bits=rand_floats(size=x.shape[0], mag=mag, minval=4., maxval=8.))),#Perte du gradient
  'Solarize': (lambda x, mag: solarize(x, thresholds=rand_floats(size=x.shape[0], mag=mag, minval=1/256., maxval=256/256.))), #Perte du gradient #=>Image entre [0,1] #Pas opti pour des batch

  #Color TF (Common mag scale)
  '+Contrast': (lambda x, mag: contrast(x, contrast_factor=rand_floats(size=x.shape[0], mag=mag, minval=1.0, maxval=1.9))),
  '+Color':(lambda x, mag: color(x, color_factor=rand_floats(size=x.shape[0], mag=mag, minval=1.0, maxval=1.9))),
  '+Brightness':(lambda x, mag: brightness(x, brightness_factor=rand_floats(size=x.shape[0], mag=mag, minval=1.0, maxval=1.9))),
  '+Sharpness':(lambda x, mag: sharpeness(x, sharpness_factor=rand_floats(size=x.shape[0], mag=mag, minval=1.0, maxval=1.9))),
  '-Contrast': (lambda x, mag: contrast(x, contrast_factor=invScale_rand_floats(size=x.shape[0], mag=mag, minval=0.1, maxval=1.0))),
  '-Color':(lambda x, mag: color(x, color_factor=invScale_rand_floats(size=x.shape[0], mag=mag, minval=0.1, maxval=1.0))),
  '-Brightness':(lambda x, mag: brightness(x, brightness_factor=invScale_rand_floats(size=x.shape[0], mag=mag, minval=0.1, maxval=1.0))),
  '-Sharpness':(lambda x, mag: sharpeness(x, sharpness_factor=invScale_rand_floats(size=x.shape[0], mag=mag, minval=0.1, maxval=1.0))),
  '=Posterize': (lambda x, mag: posterize(x, bits=invScale_rand_floats(size=x.shape[0], mag=mag, minval=4., maxval=8.))),#Perte du gradient
  '=Solarize': (lambda x, mag: solarize(x, thresholds=invScale_rand_floats(size=x.shape[0], mag=mag, minval=1/256., maxval=256/256.))), #Perte du gradient #=>Image entre [0,1] #Pas opti pour des batch
  

  'BRotate': (lambda x, mag: rotate(x, angle=rand_floats(size=x.shape[0], mag=mag, maxval=30*3))),
  'BTranslateX': (lambda x, mag: translate(x, translation=zero_stack(rand_floats(size=(x.shape[0],), mag=mag, maxval=20*3), zero_pos=0))),
  'BTranslateY': (lambda x, mag: translate(x, translation=zero_stack(rand_floats(size=(x.shape[0],), mag=mag, maxval=20*3), zero_pos=1))),
  'BShearX': (lambda x, mag: shear(x, shear=zero_stack(rand_floats(size=(x.shape[0],), mag=mag, maxval=0.3*3), zero_pos=0))),
  'BShearY': (lambda x, mag: shear(x, shear=zero_stack(rand_floats(size=(x.shape[0],), mag=mag, maxval=0.3*3), zero_pos=1))),

  'BadTranslateX': (lambda x, mag: translate(x, translation=zero_stack(rand_floats(size=(x.shape[0],), mag=mag, minval=20*2, maxval=20*3), zero_pos=0))),
  'BadTranslateX_neg': (lambda x, mag: translate(x, translation=zero_stack(rand_floats(size=(x.shape[0],), mag=mag, minval=-20*3, maxval=-20*2), zero_pos=0))),
  'BadTranslateY': (lambda x, mag: translate(x, translation=zero_stack(rand_floats(size=(x.shape[0],), mag=mag, minval=20*2, maxval=20*3), zero_pos=1))),
  'BadTranslateY_neg': (lambda x, mag: translate(x, translation=zero_stack(rand_floats(size=(x.shape[0],), mag=mag, minval=-20*3, maxval=-20*2), zero_pos=1))),
  
  'BadColor':(lambda x, mag: color(x, color_factor=rand_floats(size=x.shape[0], mag=mag, minval=1.9, maxval=2*2))),
  'BadSharpness':(lambda x, mag: sharpeness(x, sharpness_factor=rand_floats(size=x.shape[0], mag=mag, minval=1.9, maxval=2*2))),
  'BadContrast': (lambda x, mag: contrast(x, contrast_factor=rand_floats(size=x.shape[0], mag=mag, minval=1.9, maxval=2*2))),
  'BadBrightness':(lambda x, mag: brightness(x, brightness_factor=rand_floats(size=x.shape[0], mag=mag, minval=1.9, maxval=2*2))),
  
  #Non fonctionnel
  #'Auto_Contrast': (lambda mag: None), #Pas opti pour des batch (Super lent)
  #'Equalize': (lambda mag: None),
}

TF_no_mag={'Identity', 'FlipUD', 'FlipLR'}
TF_ignore_mag= TF_no_mag | {'Solarize', 'Posterize'}

def int_image(float_image): #ATTENTION : legere perte d'info (granularite : 1/256 = 0.0039)
  return (float_image*255.).type(torch.uint8)

def float_image(int_image):
  return int_image.type(torch.float)/255.

#def rand_inverse(value):
#    return  value if random.random() < 0.5 else -value

#def rand_int(mag, maxval, minval=None): #[(-maxval,minval), maxval]
#  real_max = int_parameter(mag, maxval=maxval)
#  if not minval : minval = -real_max
#  return random.randint(minval, real_max)

#def rand_float(mag, maxval, minval=None): #[(-maxval,minval), maxval]
#  real_max = float_parameter(mag, maxval=maxval)
#  if not minval : minval = -real_max
#  return random.uniform(minval, real_max)

def rand_floats(size, mag, maxval, minval=None): #[(-maxval,minval), maxval]
  real_mag = float_parameter(mag, maxval=maxval)
  if not minval : minval = -real_mag
  #return random.uniform(minval, real_max)
  return minval + (real_mag-minval) * torch.rand(size, device=mag.device) #[min_val, real_mag]

def invScale_rand_floats(size, mag, maxval, minval):
  #Mag=[0,PARAMETER_MAX] => [PARAMETER_MAX, 0] = [maxval, minval]
  real_mag = float_parameter(float(PARAMETER_MAX) - mag, maxval=maxval-minval)+minval 
  return real_mag + (maxval-real_mag) * torch.rand(size, device=mag.device) #[real_mag, max_val]

def zero_stack(tensor, zero_pos):
  if zero_pos==0:
    return torch.stack((tensor, torch.zeros((tensor.shape[0],), device=tensor.device)), dim=1)
  if zero_pos==1:
    return torch.stack((torch.zeros((tensor.shape[0],), device=tensor.device), tensor), dim=1)
  else:
    raise Exception("Invalid zero_pos : ", zero_pos) 
    
#https://github.com/tensorflow/models/blob/fc2056bce6ab17eabdc139061fef8f4f2ee763ec/research/autoaugment/augmentation_transforms.py#L137
PARAMETER_MAX = 1  # What is the max 'level' a transform could be predicted
def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .
  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.
  Returns:
    A float that results from scaling `maxval` according to `level`.
  """

  #return float(level) * maxval / PARAMETER_MAX
  return (level * maxval / PARAMETER_MAX)#.to(torch.float)

#def int_parameter(level, maxval): #Perte de gradient
  """Helper function to scale `val` between 0 and maxval .
  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.
  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  #return int(level * maxval / PARAMETER_MAX)
#  return (level * maxval / PARAMETER_MAX) 

def flipLR(x):
    device = x.device
    (batch_size, channels, h, w) = x.shape

    M =torch.tensor( [[[-1.,  0., w-1],
                        [ 0.,  1.,  0.],
                        [ 0.,  0.,  1.]]], device=device).expand(batch_size,-1,-1)

    # warp the original image by the found transform
    return kornia.warp_perspective(x, M, dsize=(h, w))

def flipUD(x):
    device = x.device
    (batch_size, channels, h, w) = x.shape

    M =torch.tensor( [[[ 1.,  0.,  0.],
                        [ 0., -1.,  h-1],
                        [ 0.,  0.,  1.]]], device=device).expand(batch_size,-1,-1)

    # warp the original image by the found transform
    return kornia.warp_perspective(x, M, dsize=(h, w))

def rotate(x, angle):
  return kornia.rotate(x, angle=angle.type(torch.float)) #Kornia ne supporte pas les int

def translate(x, translation):
  #print(translation)
  return kornia.translate(x, translation=translation.type(torch.float)) #Kornia ne supporte pas les int

def shear(x, shear):
  return kornia.shear(x, shear=shear)

def contrast(x, contrast_factor):
  return kornia.adjust_contrast(x, contrast_factor=contrast_factor) #Expect image in the range of [0, 1]

#https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageEnhance.py
def color(x, color_factor):
    (batch_size, channels, h, w) = x.shape

    gray_x = kornia.rgb_to_grayscale(x)
    gray_x = gray_x.repeat_interleave(channels, dim=1)
    return blend(gray_x, x, color_factor).clamp(min=0.0,max=1.0) #Expect image in the range of [0, 1]

def brightness(x, brightness_factor):
    device = x.device

    return blend(torch.zeros(x.size(), device=device), x, brightness_factor).clamp(min=0.0,max=1.0) #Expect image in the range of [0, 1]

def sharpeness(x, sharpness_factor):
    device = x.device
    (batch_size, channels, h, w) = x.shape

    k = torch.tensor([[[ 1.,  1.,  1.],
                       [ 1.,  5.,  1.],
                       [ 1.,  1.,  1.]]], device=device) #Smooth Filter : https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageFilter.py
    smooth_x = kornia.filter2D(x, kernel=k, border_type='reflect', normalized=True) #Peut etre necessaire de s'occuper du channel Alhpa differement

    return blend(smooth_x, x, sharpness_factor).clamp(min=0.0,max=1.0) #Expect image in the range of [0, 1]

#https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py
def posterize(x, bits):
  bits = bits.type(torch.uint8) #Perte du gradient
  x = int_image(x) #Expect image in the range of [0, 1]

  mask = ~(2 ** (8 - bits) - 1).type(torch.uint8)

  (batch_size, channels, h, w) = x.shape
  mask = mask.unsqueeze(dim=1).expand(-1,channels).unsqueeze(dim=2).expand(-1,channels, h).unsqueeze(dim=3).expand(-1,channels, h, w) #Il y a forcement plus simple ...

  return float_image(x & mask)

def auto_contrast(x): #PAS OPTIMISE POUR DES BATCH #EXTRA LENT
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

def equalize(x): #PAS OPTIMISE POUR DES BATCH
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

def solarize(x, thresholds): #PAS OPTIMISE POUR DES BATCH
  # Optimisation : Mask direct sur toute les donnees (Mask = (B,C,H,W)> (B))
  batch_size, channels, h, w = x.shape
  imgs=[]
  for idx, t in enumerate(thresholds): #Operation par image
    mask = x[idx] > t #Perte du gradient
    #In place
    inv_x = 1-x[idx][mask]
    x[idx][mask]=inv_x
    #

  #Out of place
  #  im = x[idx]
  #  inv_x = 1-im[mask]

  #  imgs.append(im.masked_scatter(mask,inv_x))

  #idxs=torch.tensor(range(x.shape[0]), device=x.device)
  #idxs=idxs.unsqueeze(dim=1).expand(-1,channels).unsqueeze(dim=2).expand(-1,channels, h).unsqueeze(dim=3).expand(-1,channels, h, w) #Il y a forcement plus simple ...
  #x=x.scatter(dim=0, index=idxs, src=torch.stack(imgs))
  #
  return x

#https://github.com/python-pillow/Pillow/blob/9c78c3f97291bd681bc8637922d6a2fa9415916c/src/PIL/Image.py#L2818
def blend(x,y,alpha): #out = image1 * (1.0 - alpha) + image2 * alpha
    #return kornia.add_weighted(src1=x, alpha=(1-alpha), src2=y, beta=alpha, gamma=0) #out=src1∗alpha+src2∗beta+gamma #Ne fonctionne pas pour des batch de alpha

    if not isinstance(x, torch.Tensor):
        raise TypeError("x should be a tensor. Got {}".format(type(x)))

    if not isinstance(y, torch.Tensor):
        raise TypeError("y should be a tensor. Got {}".format(type(y)))

    (batch_size, channels, h, w) = x.shape
    alpha = alpha.unsqueeze(dim=1).expand(-1,channels).unsqueeze(dim=2).expand(-1,channels, h).unsqueeze(dim=3).expand(-1,channels, h, w) #Il y a forcement plus simple ...
    res = x*(1-alpha) + y*alpha

    return res
