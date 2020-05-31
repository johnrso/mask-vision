from xml.etree import ElementTree
import os
import shutil
import cv2
import random
import progressbar

imagePath = "./medical-mask-set/images/"
labelPath = "./medical-mask-set/labels/"

paths = ["./cropped/",
         "./cropped/good/",
         "./cropped/bad/",
         "./cropped/none/"]

data = ["./data/",
         "./data/train/",
         "./data/test/",
         "./data/train/good/",
         "./data/train/bad/",
         "./data/train/none/",
         "./data/test/good/",
         "./data/test/bad/",
         "./data/test/none/"]

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

        #files incorrectly labeled?
        if label == "bad":
            label = "none"
        elif label == "none":
            label = "bad"
        try:
            filepath = os.path.join(paths[0], label, name + str(ind) + ".jpeg")
            cv2.imwrite(filepath, image)
            ind += 1
        except CV2.error:
            print("image unsuccessfully saved.")

def cropImages():
    print("extracting faces.")
    for p in paths:
        if not os.path.exists(p):
            try:
                os.mkdir(p)
            except OSError:
                print("error encountered while creating " + p)

    widgets = [progressbar.Timer(format='elapsed time: %(elapsed)s',),
               " ", progressbar.AnimatedMarker(markers='.oO@* ')]
    bar = progressbar.ProgressBar(widgets=widgets)

    for image in os.listdir(imagePath):
        processImage(os.path.join(imagePath, image), labelPath)
        bar.update()
    print()
    print()
    for p in paths[1:]:
        print(p + ": " + str(len(os.listdir(p))) + " images.")

    print()
    print("finished extracting faces.")
    print()

def splitData():
    print("splitting data.")
    for p in data:
        if not os.path.exists(p):
            try:
                os.mkdir(p)
            except OSError:
                print("error encountered while creating " + p)

    train_test = [.8, .2]

    widgets = [progressbar.Timer(format='elapsed time: %(elapsed)s',),
                   " ", progressbar.AnimatedMarker(markers='.oO@* ')]
    bar = progressbar.ProgressBar(widgets=widgets)


    for path in paths[1:]:
        dir = os.listdir(path)
        name = os.path.basename(os.path.dirname(path))
        split = int(train_test[1] * len(dir))
        test_split = random.choices(dir, k = split)
        for file in dir:
            orig = os.path.join(path, file)
            if file not in test_split:
                filepath = os.path.join(data[1], name)
            else:
                filepath = os.path.join(data[2], name)
            shutil.copy(orig, filepath)
            bar.update()

    print()
    print()
    for p in data[3:]:
        print(p + ": " + str(len(os.listdir(p))) + " images.")

    print()
    print("finished splitting data.")
    print()

def clearDir():
    print("\nclearing existing data.")
    shutil.rmtree(paths[0], ignore_errors = True)
    shutil.rmtree(data[0], ignore_errors = True)
    print()

clearDir()
cropImages()
splitData()
