import torch
import cv2
import numpy as np
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from structureNN import establish_device, NeuralNetwork
from matplotlib import pyplot as plt

def show_tensor_image(img):
    img_array = img.numpy()
    imgNormalized = cv2.normalize(img_array,None, 0, 255, cv2.NORM_MINMAX)
    imgFinal = np.array(np.round(imgNormalized.reshape((28,28))), dtype=np.uint8)

    print(imgFinal)
    print(imgFinal.shape)
    #cv2.imwrite("a.bmp", test_data[0][0])
    plt.imshow(imgFinal, cmap='gray')
    plt.show()

def putBorder(image, backgroundColor,border):
    (h,w) = image.shape[:2]
    mask = np.full((h+border,w+border), backgroundColor, dtype = image.dtype)
    b2 = border//2
    if len(img.shape)==2:
        mask[b2:h+b2,b2:w+b2] = image[:h,:w]
    else: 
        mask[b2:h+b2,b2:w+b2] = image[:h,:w,:]
    return mask

def resize_image(image,backgroundColor,size):
    border = putBorder(image,backgroundColor,10)

    (h,w) = border.shape[:2]
    maior = h if h>w else w

    x_pos = (maior - w)//2
    y_pos = (maior - h)//2

    mask = np.full((maior,maior), backgroundColor, dtype = border.dtype)
    if len(img.shape) == 2:
        mask[y_pos:y_pos+h,x_pos:x_pos+w] = border[:h,:w]
    else:
        mask[y_pos:y_pos+h,x_pos:x_pos+w] = border[:h,:w,:]
        
    return cv2.resize(mask,(size,size),interpolation=cv2.INTER_AREA)

def treat_image(img):
    image = ~img
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
    #image = cv2.bitwise_and(image,image, mask=thresh)
    return resize_image(image, 0, 28)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


test_data = datasets.FashionMNIST(
    root = 'data',
    train = False,
    download = True,
    transform = ToTensor()
)

device = establish_device()
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

for i,a in enumerate(test_data):
    if(a[1] == 3):
        idx = i
        break


show_tensor_image(test_data[idx][0])
model.eval()
x, y = test_data[idx][0], test_data[idx][1]
with torch.no_grad():
    x = x.to(device)
    pred=model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f"Predicted: '{predicted}', Actual: '{actual}'")


img = cv2.imread("shirt.jpeg", cv2.IMREAD_GRAYSCALE)
img = treat_image(img)

#img = NormalizeData(np.float32(img))

transform = transforms.Compose([
    transforms.ToTensor()
])
tensor_image = transform(img)

show_tensor_image(tensor_image)

model.eval()
x = tensor_image
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted = classes[pred[0].argmax(0)]
    print(f'Predicted: "{predicted}"\n')
    print(f"Predicition:")
    for i, values in enumerate(pred[0]):
        print(f"{classes[i]}: {values}")