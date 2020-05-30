from xml.etree import ElementTree
import os
import shutil
import cv2

# image splitting inspired by

def imageToXML(imagePath, labelDir):
    name = os.path.splitext(os.path.basename(imagePath))[0] + ".xml"
    return os.path.join(labelDir, name)

def splitImage(image, labelDir):
    xml = imageToXML(image, labelDir)
    root = ElementTree.parse(xml).getroot()
    regions = []
    for object_tag in root.findall("object"):
        label = object_tag.find("name").text
        xmin = int(object_tag.find("bndbox/xmin").text)
        xmax = int(object_tag.find("bndbox/xmax").text)
        ymin = int(object_tag.find("bndbox/ymin").text)
        ymax = int(object_tag.find("bndbox/ymax").text)
        regions.append({ "label": label,
                         "coord": (xmin, ymin, xmax, ymax) })
    return regions

def cropFace(img, coord, dim = (50,50)):
    cropped = img[coord[1] : coord[3], coord[0] : coord[2]]
    return cv2.resize(cropped, dim)

#todo: denoising
def processImage(image, labelDir):
    full = cv2.imread(image)
    name = os.path.splitext(os.path.basename(image))[0] + ".xml"
    regions = splitImage(image, labelDir)

    ind = 0
    for ROI in regions:
        label = ROI["label"]
        image = cropFace(full, ROI["coord"])

        try:
            filepath = os.path.join(paths[0], label, name + str(ind) + ".jpeg")
            cv2.imwrite(filepath, image)
            ind += 1
        except CV2.error:
            print("image unsuccessfully saved.")

def splitImages():

imagePath = "./medical-mask-set/images/"
labelPath = "./medical-mask-set/labels/"

paths = ["./cropped/",
         "./cropped/good/",
         "./cropped/bad/",
         "./cropped/none/"]

for p in paths:
    if not os.path.exists(p):
        try:
            os.mkdir(p)
        except OSError:
            print("error encountered while creating " + p)

for image in os.listdir(imagePath):
    processImage(os.path.join(imagePath, image), labelPath)

goodImg = len(os.listdir(paths[1]))
badImg = len(os.listdir(paths[2]))
noneImg = len(os.listdir(paths[3]))

print("total number of good faces: " + str(goodImg))
print("total number of bad faces: " + str(badImg))
print("total number of no masks: " + str(noneImg))
print()
print("finished extracting features. now spliting data.")

data = ["./data/",
         "./data/train/",
         "./data/train/good/",
         "./data/train/bad/",
         "./data/train/none/",
         "./data/test/",
         "./data/test/good/",
         "./data/test/bad/",
         "./data/test/none/",
         "./data/val/",
         "./data/val/good/",
         "./data/val/bad/",
         "./data/val/none/",]

for p in data:
    if not os.path.exists(p):
        try:
            os.mkdir(p)
        except OSError:
            print("error encountered while creating " + p)

train_test_val = [.7, .2, .1]
