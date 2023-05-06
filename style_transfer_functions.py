import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from  PIL import Image, ImageEnhance
import time
import os
import numpy as np
from Models.ST_VAE.libs.models import encoder4
from Models.ST_VAE.libs.models import decoder4
from Models.ST_VAE.libs.Matrix import MulLayer


class VAE():
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
            transforms.Lambda(lambda x: x[:3])
        ])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        vgg = encoder4()
        dec = decoder4()
        matrix = MulLayer(z_dim=256)
        vgg.load_state_dict(torch.load(   "Models/ST_VAE/models/vgg_r41.pth", map_location=torch.device(self.device)))
        dec.load_state_dict(torch.load(   "Models/ST_VAE/models/dec_r41.pth", map_location=torch.device(self.device)))
        matrix.load_state_dict(torch.load("Models/ST_VAE/models/matrix_r41_new.pth", map_location=torch.device(self.device)))

        vgg.to(self.device)
        dec.to(self.device)
        matrix.to(self.device)
        matrix.eval()
        vgg.eval()
        dec.eval()
        self.vgg = vgg
        self.dec = dec
        self.matrix = matrix

    def transform_image(self, content, ref):

        content = self.transform(content).unsqueeze(0).to(self.device)
        ref = self.transform(ref).unsqueeze(0).to(self.device)
        # print(content.shape, ref.shape)

        with torch.no_grad():
            sF = self.vgg(ref)
            cF = self.vgg(content)
            feature, _, _ = self.matrix(cF['r41'], sF['r41'])
            prediction = self.dec(feature)

            prediction = prediction.data[0].cpu().permute(1, 2, 0)

        # t1 = time.time()
        #print("===> Processing: %s || Timer: %.4f sec." % (str(i), (t1 - t0)))

        prediction = prediction * 255.0
        prediction = prediction.clamp(0, 255)


        transformed_img = Image.fromarray(np.uint8(prediction))
        return transformed_img

