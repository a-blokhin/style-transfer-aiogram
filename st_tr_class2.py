import numpy as np
from PIL import Image
import copy
from aiogram import executor
import asyncio


from loader import event_loop


from torchvision.models import vgg19
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt


from torchvision import transforms
from io import BytesIO

import copy
'''Функции потерь и функция нормализации'''
class ContentLoss(nn.Module):

        def __init__(self, target,):
            super(ContentLoss, self).__init__()
            # we 'detach' the target content from the tree used
            # to dynamically compute the gradient: this is a stated value,
            # not a variable. Otherwise the forward method of the criterion
            # will throw an error.
            self.target = target.detach()#это константа. Убираем ее из дерева вычеслений
            self.loss = F.mse_loss(self.target, self.target )#to initialize with something

        def forward(self, input):
            self.loss = F.mse_loss(input, self.target)
            return input

class StyleLoss(nn.Module):
        def __init__(self, target_feature):
            super(StyleLoss, self).__init__()
            self.target = self.gram_matrix(target_feature).detach()
            self.loss = F.mse_loss(self.target, self.target)# to initialize with something

        def gram_matrix(self, input):
          batch_size , h, w, f_map_num = input.size()  # batch size(=1)
          # b=number of feature maps
          # (h,w)=dimensions of a feature map (N=h*w)

          features = input.view(batch_size * h, w * f_map_num)  # resise F_XL into \hat F_XL

          G = torch.mm(features, features.t())  # compute the gram product

          # we 'normalize' the values of the gram matrix
          # by dividing by the number of element in each feature maps.
          return G.div(batch_size * h * w * f_map_num)

        def forward(self, input):
            G = self.gram_matrix(input)
            self.loss = F.mse_loss(G, self.target)
            return input

class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            # .view the mean and std to make them [C x 1 x 1] so that they can
            # directly work with image Tensor of shape [B x C x H x W].
            # B is batch size. C is number of channels. H is height and W is width.
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

        def forward(self, img):
            # normalize img
            return (img - self.mean) / self.std
        
'''Класс переноса стиля самы простой на vgg19,сложные картинки обсчитывается 5-10 минут в зависимости от cpu, сначла все картинки сжимаем до 200*200 потом раскрываем до 1024*1024, получается неплохо при наложении фильтров в виде картин,'''

class run_style_transfer(object):

      def image_loader(self, image_name):
            loader = transforms.Compose([
                transforms.Resize(200),  # нормируем размер изображения
                transforms.CenterCrop(200),
                transforms.ToTensor()])  # превращаем в удобный формат


            image = loader(image_name).unsqueeze(0)
            return image.to(torch.float)

      def get_input_optimizer(self, input_img):
              # this line to show that input is a parameter that requires a gradient
              #добоваляет содержимое тензора катринки в список изменяемых оптимизатором параметров
              optimizer = optim.LBFGS([input_img.requires_grad_()])
              return optimizer

      def get_style_model_and_losses(self,
              style_img, content_img,
              cnn,
              normalization_mean = torch.tensor([0.485, 0.456, 0.406]),
              normalization_std = torch.tensor([0.229, 0.224, 0.225]),
              content_layers=['conv_4'],
              style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):

                  cnn = copy.deepcopy(cnn)

                  # normalization module
                  normalization = Normalization(normalization_mean, normalization_std)

                  # just in order to have an iterable access to or list of content/syle
                  # losses
                  content_losses = []
                  style_losses = []

                  # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
                  # to put in modules that are supposed to be activated sequentially
                  model = nn.Sequential(normalization)

                  i = 0  # increment every time we see a conv
                  for layer in cnn.children():
                      if isinstance(layer, nn.Conv2d):
                          i += 1
                          name = 'conv_{}'.format(i)
                      elif isinstance(layer, nn.ReLU):
                          name = 'relu_{}'.format(i)
                          # The in-place version doesn't play very nicely with the ContentLoss
                          # and StyleLoss we insert below. So we replace with out-of-place
                          # ones here.
                          #Переопределим relu уровень
                          layer = nn.ReLU(inplace=False)
                      elif isinstance(layer, nn.MaxPool2d):
                          name = 'pool_{}'.format(i)
                      elif isinstance(layer, nn.BatchNorm2d):
                          name = 'bn_{}'.format(i)
                      else:
                          raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

                      model.add_module(name, layer)

                      if name in content_layers:
                          # add content loss:
                          target = model(content_img).detach()
                          content_loss = ContentLoss(target)
                          model.add_module("content_loss_{}".format(i), content_loss)
                          content_losses.append(content_loss)

                      if name in style_layers:
                          # add style loss:
                          target_feature = model(style_img).detach()
                          style_loss = StyleLoss(target_feature)
                          model.add_module("style_loss_{}".format(i), style_loss)
                          style_losses.append(style_loss)

                  # now we trim off the layers after the last content and style losses
                  #выбрасываем все уровни после последенего styel loss или content loss
                  for i in range(len(model) - 1, -1, -1):
                      if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                          break

                  model = model[:(i + 1)]

                  return model, style_losses, content_losses

      async def run_style_transfer(self,
              content_img, style_img,
              cnn = vgg19(pretrained=True).features.eval(),
              num_steps=500,
              style_weight=100000, content_weight=1, loop=event_loop):


                  """Run the style transfer."""
                  content_img = self.image_loader(content_img)
                  style_img = self.image_loader(style_img)
                  input_img = content_img.clone()
                  content_img.requires_grad = False
                  style_img.requires_grad = False






                  model, style_losses, content_losses = self.get_style_model_and_losses(style_img, content_img, cnn)
                  optimizer = self.get_input_optimizer(input_img)

                  run = [0]
                  style_scores = []
                  while run[0] <= num_steps:
                      await asyncio.sleep(1)

                      def closure():
                          # correct the values
                          # это для того, чтобы значения тензора картинки не выходили за пределы [0;1]
                          input_img.data.clamp_(0, 1)

                          optimizer.zero_grad()

                          model(input_img)

                          style_score = 0
                          content_score = 0

                          for sl in style_losses:
                              style_score += sl.loss
                          for cl in content_losses:
                              content_score += cl.loss

                          #взвешивание ощибки
                          style_score *= style_weight
                          content_score *= content_weight

                          loss = style_score + content_score
                          loss.backward()


                          run[0] += 1
                          if run[0] % 50 == 0:
                              print(run[0])
                              style_scores.append(style_score.item())


                          return style_score + content_score

                      if run[0]>=100 and (style_scores[-2:-1][0] - style_scores[-1:][0] < 1):
                        break



                      optimizer.step(closure)

                  # a last correction...
                  byte_input = BytesIO()
                  input_img.data.clamp_(0, 1)
                  res = transforms.Resize(1024)
                  input_img = res(input_img.detach())
                  input_img = input_img.squeeze(0).permute(1,2,0).detach().numpy()
                  input_img = Image.fromarray((input_img * 255).astype('uint8'))
                  input_img.save(byte_input, 'JPEG')
                  input_img = byte_input.getvalue()
                  return input_img

