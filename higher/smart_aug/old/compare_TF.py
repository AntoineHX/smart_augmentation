import RandAugment as rand
import PIL
import torchvision
import transformations as TF
tpil=torchvision.transforms.ToPILImage()
ttensor=torchvision.transforms.ToTensor()

img,label =data_train[0]

rimg=ttensor(PIL.ImageEnhance.Color(tpil(img)).enhance(1.5))#ttensor(PIL.ImageOps.solarize(tpil(img), 50))#ttensor(tpil(img).transform(tpil(img).size, PIL.Image.AFFINE, (1, -0.1, 0, 0, 1, 0)))#rand.augmentations.FlipUD(tpil(img),1))
timg=TF.color(img.unsqueeze(0),torch.Tensor([1.5])).squeeze(0)
print(torch.allclose(rimg,timg, atol=1e-3))
tpil(rimg).save('rimg.jpg')
tpil(timg).save('timg.jpg')