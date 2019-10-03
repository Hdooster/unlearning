import datetime
import os
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD


def color_MNIST(gray_dataset, colors=None, colors_rgb=None, bias=0, color_noise=False):
    ''' this function takes single-channel np.uint8 0-255 images and their labels,
    and returns RGB np.uint8 0-255 images and their colors, optionally correlating colors to the labels.
    The goal of this function is to introduce bias into the MNIST dataset.'''

    x = []
    y_color = []

    for img, label in gray_dataset:
        if np.random.rand() < bias:
            i = label
        else:
            i = random.choice(range(10))

        icolor = colors[i]
        icolor_rgb = colors_rgb[i]
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cimg = cimg.astype(np.float32)/255.

        # convert color
        cimg[..., 0] = cimg[..., 0] * icolor_rgb[0]
        cimg[..., 1] = cimg[..., 1] * icolor_rgb[1]
        cimg[..., 2] = cimg[..., 2] * icolor_rgb[2]
        cimg = cimg.astype(np.uint8)
        x.append(cimg)
        y_color.append(icolor)

    return np.array(x), np.array(y_color)


def evaluate_results(y_true, y_pred, all_labels):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=all_labels, yticklabels=all_labels,
           ylim=(len(all_labels)-0.5, -0.5),
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.show()

    # Compute accuracy
    acc = np.mean(y_true==y_pred)
    print(f'Accuracy: {acc*100.:.2f}%')
    print('Per class statistics:')
    # Compute statistics per defect type
    for i, label in enumerate(all_labels):
        totp = np.sum(cm[:, i])
        realp = np.sum(cm[i])
        tp = cm[i][i]
        fp = totp - tp
        fn = realp - tp
        prec = tp / (tp+fp)
        rec = tp / realp
        print(f'  - {label}: precision {prec*100.:.2f}%, recall {rec*100.:.2f}%')


def log_save_callbacks(name, log=True, save=True):
    # set up callbacks
    callbacks = []
    save_dir_root = "../model_training/"

    if save:
        fp_modelcheckpoint = os.path.join(save_dir_root, "modelcheckpoints")
        os.makedirs(fp_modelcheckpoint, exist_ok=True)
        h5_filename = os.path.join(fp_modelcheckpoint, name + ".hdf5")
        callbacks += [ModelCheckpoint(h5_filename, save_best_only=True)]

    if log:
        dir_tensorboard = os.path.join(save_dir_root, "tensorboard", name)
        os.makedirs(dir_tensorboard, exist_ok=True)
        callbacks += [TensorBoard(dir_tensorboard)]

    return callbacks


def compile_model(model, name, loss, loss_weights=None, initial_lr=1e-5, patience=10):

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    callbacks = log_save_callbacks(name=name + '_' + timestamp, log=True, save=False)
    callbacks += [EarlyStopping(patience=patience, verbose=1, restore_best_weights=True)]
    callbacks += [ReduceLROnPlateau(verbose=1, factor=0.2, patience=int(.75*patience))]

    optimizer = SGD(initial_lr)
    model.compile(optimizer, loss=loss, metrics=['acc'], loss_weights=loss_weights)
    return model, callbacks
        
colors = dict([
    ["maroon", (128,0,0)],
    ["dark red", (139,0,0)],
    ["brown", (165,42,42)],
    ["firebrick", (178,34,34)],
    ["crimson", (220,20,60)],
    ["red", (255,0,0)],
    ["tomato", (255,99,71)],
    ["coral", (255,127,80)],
    ["indian red", (205,92,92)],
    ["light coral", (240,128,128)],
    ["dark salmon", (233,150,122)],
    ["salmon", (250,128,114)],
    ["light salmon", (255,160,122)],
    ["orange red", (255,69,0)],
    ["dark orange", (255,140,0)],
    ["orange", (255,165,0)],
    ["gold", (255,215,0)],
    ["dark golden rod", (184,134,11)],
    ["golden rod", (218,165,32)],
    ["pale golden rod", (238,232,170)],
    ["dark khaki", (189,183,107)],
    ["khaki", (240,230,140)],
    ["olive", (128,128,0)],
    ["yellow", (255,255,0)],
    ["yellow green", (154,205,50)],
    ["dark olive green", (85,107,47)],
    ["olive drab", (107,142,35)],
    ["lawn green", (124,252,0)],
    ["chart reuse", (127,255,0)],
    ["green yellow", (173,255,47)],
    ["dark green", (0,100,0)],
    ["green", (0,128,0)],
    ["forest green", (34,139,34)],
    ["lime", (0,255,0)],
    ["lime green", (50,205,50)],
    ["light green", (144,238,144)],
    ["pale green", (152,251,152)],
    ["dark sea green", (143,188,143)],
    ["medium spring green", (0,250,154)],
    ["spring green", (0,255,127)],
    ["sea green", (46,139,87)],
    ["medium aqua marine", (102,205,170)],
    ["medium sea green", (60,179,113)],
    ["light sea green", (32,178,170)],
    ["dark slate gray", (47,79,79)],
    ["teal", (0,128,128)],
    ["dark cyan", (0,139,139)],
    ["aqua", (0,255,255)],
    ["cyan", (0,255,255)],
    ["light cyan", (224,255,255)],
    ["dark turquoise", (0,206,209)],
    ["turquoise", (64,224,208)],
    ["medium turquoise", (72,209,204)],
    ["pale turquoise", (175,238,238)],
    ["aqua marine", (127,255,212)],
    ["powder blue", (176,224,230)],
    ["cadet blue", (95,158,160)],
    ["steel blue", (70,130,180)],
    ["corn flower blue", (100,149,237)],
    ["deep sky blue", (0,191,255)],
    ["dodger blue", (30,144,255)],
    ["light blue", (173,216,230)],
    ["sky blue", (135,206,235)],
    ["light sky blue", (135,206,250)],
    ["midnight blue", (25,25,112)],
    ["navy", (0,0,128)],
    ["dark blue", (0,0,139)],
    ["medium blue", (0,0,205)],
    ["blue", (0,0,255)],
    ["royal blue", (65,105,225)],
    ["blue violet", (138,43,226)],
    ["indigo", (75,0,130)],
    ["dark slate blue", (72,61,139)],
    ["slate blue", (106,90,205)],
    ["medium slate blue", (123,104,238)],
    ["medium purple", (147,112,219)],
    ["dark magenta", (139,0,139)],
    ["dark violet", (148,0,211)],
    ["dark orchid", (153,50,204)],
    ["medium orchid", (186,85,211)],
    ["purple", (128,0,128)],
    ["thistle", (216,191,216)],
    ["plum", (221,160,221)],
    ["violet", (238,130,238)],
    ["magenta / fuchsia", (255,0,255)],
    ["orchid", (218,112,214)],
    ["medium violet red", (199,21,133)],
    ["pale violet red", (219,112,147)],
    ["deep pink", (255,20,147)],
    ["hot pink", (255,105,180)],
    ["light pink", (255,182,193)],
    ["pink", (255,192,203)],
    ["antique white", (250,235,215)],
    ["beige", (245,245,220)],
    ["bisque", (255,228,196)],
    ["blanched almond", (255,235,205)],
    ["wheat", (245,222,179)],
    ["corn silk", (255,248,220)],
    ["lemon chiffon", (255,250,205)],
    ["light golden rod yellow", (250,250,210)],
    ["light yellow", (255,255,224)],
    ["saddle brown", (139,69,19)],
    ["sienna", (160,82,45)],
    ["chocolate", (210,105,30)],
    ["peru", (205,133,63)],
    ["sandy brown", (244,164,96)],
    ["burly wood", (222,184,135)],
    ["tan", (210,180,140)],
    ["rosy brown", (188,143,143)],
    ["moccasin", (255,228,181)],
    ["navajo white", (255,222,173)],
    ["peach puff", (255,218,185)],
    ["misty rose", (255,228,225)],
    ["lavender blush", (255,240,245)],
    ["linen", (250,240,230)],
    ["old lace", (253,245,230)],
    ["papaya whip", (255,239,213)],
    ["sea shell", (255,245,238)],
    ["mint cream", (245,255,250)],
    ["slate gray", (112,128,144)],
    ["light slate gray", (119,136,153)],
    ["light steel blue", (176,196,222)],
    ["lavender", (230,230,250)],
    ["floral white", (255,250,240)],
    ["alice blue", (240,248,255)],
    ["ghost white", (248,248,255)],
    ["honeydew", (240,255,240)],
    ["ivory", (255,255,240)],
    ["azure", (240,255,255)],
    ["snow", (255,250,250)],
    # ["black", (0,0,0)],
    ["dim gray / dim grey", (105,105,105)],
    ["gray / grey", (128,128,128)],
    ["dark gray / dark grey", (169,169,169)],
    ["silver", (192,192,192)],
    ["light gray / light grey", (211,211,211)],
    ["gainsboro", (220,220,220)],
    ["white smoke", (245,245,245)],
    ["white", (255,255,255)]
])
