import torch
from model import TransformerNet
import argparse
import cv2

from torchvision import transforms

def load_image(path):
    img = cv2.imread(path)
    return img


def save_image(img, path):
    img = img.clip(0, 255)
    return cv2.imwrite(path, img)



def itoi(img, max_size = None):
    if max_size == None:
        tfs = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.mul(255))
        ])

    else:
        H, W, C = img.shape
        img_size = tuple([int((float(max_size) / max(H, W))*x) for x in [H, W]])
        tfs = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize(img_size),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.mul(255))
        ])

    return tfs(img)


def toti(tensor):
    tensor = tensor.squeeze()
    img = tensor.cpu().numpy()
    return img.transpose(1, 2, 0)


def stylize(content_image, style_path, out):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TransformerNet().to(device)
    model.load_state_dict(torch.load(style_path, map_location = device))

    with torch.no_grad():
        torch.cuda.empty_cache()
        #content_image = load_image(path)
        content_tensor = itoi(content_image, 512).unsqueeze(0).to(device)
        gen_tensor = model(content_tensor)
        gen_image = toti(gen_tensor)
        _, a = cv2.imencode('.jpg', gen_image, )
        save_image(gen_image, out)
        return a
