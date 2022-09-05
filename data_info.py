import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# flevo_c
flevo_c_colors = np.zeros([4, 3])
flevo_c_colors[0, :] = [0, 1, 254] #water
flevo_c_colors[1, :] = [0, 131, 71] #forest
flevo_c_colors[2, :] = [253, 0, 0] #urban
flevo_c_colors[3, :] = [0, 253, 255] #cropland

flevo_c_names = ["water", "forest", "urban", "cropland"]
flevo_c_classes = {"water": 0, "forest": 1, "urban": 2, "cropland": 3}
flevo_c_y = pd.read_pickle("../y/flevo_c.pkl")
flevo_c_img = plt.imread("pics/flevo_c.jpg")


#flevo_l
flevo_l_colors = np.zeros([15, 3])
flevo_l_colors[0, :] = [0, 1, 254] #water
flevo_l_colors[1, :] = [0, 131, 71] #forest
flevo_l_colors[2, :] = [0, 253, 255] #lucerne
flevo_l_colors[3, :] = [0, 255, 0] #grass
flevo_l_colors[4, :] = [255, 126, 0] #rapessed
flevo_l_colors[5, :] = [180, 0, 255] #beet
flevo_l_colors[6, :] = [251, 255, 7] #potates
flevo_l_colors[7, :] = [91, 8, 227] #peas
flevo_l_colors[8, :] = [253, 0, 0] #steam beans
flevo_l_colors[9, :] = [172, 138, 78] #bare soil
flevo_l_colors[10, :] = [255, 181, 230] #wheat A
flevo_l_colors[11, :] = [191, 191, 255] #wheat B
flevo_l_colors[12, :] = [201, 222, 188] #wheat C
flevo_l_colors[13, :] = [127, 21, 25] #Barley
flevo_l_colors[14, :] = [249, 226, 150] #Building

flevo_l_names = ["water", "forest", "lucerne", "grass", "rapessed", "beet", "potates", "peas", "steam beans", "bare soil", "wheat A", "wheat B", "wheat C", "Barley", "Building"]
flevo_l_classes = {"water": 0, "forest": 1, "lucerne": 2, "grass": 3, "rapessed": 4, "beet": 5, "potates": 6, "peas": 7, "steam beans": 8, "bare soil": 9, "wheat A": 10, "wheat B": 11, "wheat C": 12, "Barley": 13, "Building": 14}
flevo_l_y = pd.read_pickle("../y/flevo_l.pkl")
flevo_l_img = plt.imread("pics/flevo_l.jpg")

# sfbay_c
sfbay_c_colors = np.zeros([5, 3])
sfbay_c_colors[0, :] = [57, 83, 160] #water
sfbay_c_colors[1, :] = [232, 33, 38] #high urban
sfbay_c_colors[2, :] = [185, 79, 151] #developed
sfbay_c_colors[3, :] = [255, 97, 105] #low urban
sfbay_c_colors[4, :] = [104, 192, 70] #vegetation

sfbay_c_names = ["water", "high urban", "developed", "low urban", "vegetation"]
sfbay_c_classes = {"water": 0, "high urban": 1, "developed": 2, "low urban": 3, "vegetation": 4}
sfbay_c_y = pd.read_pickle("../y/sfbay_c.pkl")
sfbay_c_img = plt.imread("pics/sfbay_c.jpg")

# sfbay_l
sfbay_l_colors = np.zeros([5, 3])
sfbay_l_colors[0, :] = [57, 83, 160] #water
sfbay_l_colors[1, :] = [1, 131, 71] #urban
sfbay_l_colors[2, :] = [112, 203, 221] #forest
sfbay_l_colors[3, :] = [107, 186, 69] #bare soil
sfbay_l_colors[4, :] = [249, 126, 30] #vegetation

sfbay_l_names = ["water", "urban", "forest", "bare soil", "vegetation"]
sfbay_l_classes = {"water": 0, "urban": 1, "forest": 2, "bare soil": 3, "vegetation": 4}
sfbay_l_y = pd.read_pickle("../y/sfbay_l.pkl")
sfbay_l_img = plt.imread("pics/sfbay_l.jpg")

flevo_c = {"colors": flevo_c_colors, "classes": flevo_c_classes, "names": flevo_c_names, "y": flevo_c_y, "img": flevo_c_img}
flevo_l = {"colors": flevo_l_colors, "classes": flevo_l_classes, "names": flevo_l_names, "y": flevo_l_y, "img": flevo_l_img}
sfbay_c = {"colors": sfbay_c_colors, "classes": sfbay_c_classes, "names": sfbay_c_names, "y": sfbay_c_y, "img": sfbay_c_img}
sfbay_l = {"colors": sfbay_l_colors, "classes": sfbay_l_classes, "names": sfbay_l_names, "y": sfbay_l_y, "img": sfbay_l_img}

def mask(data, names):
    """
    It takes an image and a list of classes and returns a new image where all pixels that are not in the
    list of classes are replaced with the original image
    
    :param data: a dictionary containing the following keys:
    :param names: list of names of classes to be masked
    :return: A numpy array of the same shape as the input image, but with the colors of the classes
    masked.
    """

    img = data["img"]
    #list of all colors
    color = data["colors"]
    classes = data["classes"]
    y = data["y"]

    #colors to be masked
    colors = [classes[x] for x in names]
    
    masked_img = np.zeros(img.shape)
    counter = 0
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if y[counter] in colors:
                masked_img[i, j, :] = color[y[counter], :]
            else:
                masked_img[i, j, :] = img[i, j, :]
            counter = counter + 1
    
    color_mask = masked_img.astype(np.uint8)
    return color_mask
