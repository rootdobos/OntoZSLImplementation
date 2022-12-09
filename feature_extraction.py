import tensorflow as tf
import numpy as np
import itertools
import matplotlib.pyplot as plt
import os
#https://pyimagesearch.com/2020/04/27/fine-tuning-resnet-with-keras-tensorflow-and-deep-learning/
#https://www.tensorflow.org/tutorials/images/transfer_learning

def instantiate_model():

    IMG_SIZE = (256, 256) +(3,)

    base_model= tf.keras.applications.ResNet101V2(input_shape=IMG_SIZE, include_top=False,weights='imagenet')
    base_model.trainable=False
    headModel= base_model.output
    headModel= tf.keras.layers.AveragePooling2D(pool_size=(5,5))(headModel)
    headModel= tf.keras.layers.Flatten(name="flatten")(headModel)
    
    model= tf.keras.Model(inputs=base_model.input,outputs=headModel)


    return model
def get_entities_in_file_line(filename):
    lines=[]
    with open(filename,"r") as f:
        lines = [line.rstrip('\n') for line in f]
    return lines


def get_wordEmbeddingDictionary():
    name_id_dictionary={}
    #fileEntitiesID= open("TransE_Meta/entity_to_id.tsv","r")
    lines=[]
    with open("TransE_Meta\\training_triples\\entity_to_id.tsv","r") as f:
        lines = [line.rstrip('\n') for line in f]
    splitted=[x.split('\t') for x in lines]

    classes=[]
    with open("classes.txt","r") as f:
        classes = [line.rstrip('\n') for line in f]

    for element in splitted:
        converted="+".join(element[1].split("_"))
        if converted in classes:
            name_id_dictionary[converted]=element[0]
    lines=[]
    with open("wordEmbedding.txt","r") as f:
        lines = [line.rstrip('\n') for line in f]
    splittedVectors=[l.split(';') for l in lines]
    convertedVectors=[np.array(line).astype(np.float) for line in splittedVectors]

    name_vec_dictionary={}
    for key,value in name_id_dictionary.items():
        name_vec_dictionary[key]=convertedVectors[int(value)]
    return name_vec_dictionary

def classifier_model(output_classes):
    i=tf.keras.layers.Input(shape=2048)
    x= tf.keras.layers.Dense(512,activation='relu')(i)
    x= tf.keras.layers.Dense(output_classes,activation='softmax')(x)
    model= tf.keras.Model(i,x)
    return model


def plot_confusion_matrix(cm, classes, normalize=False, cmap=plt.cm.Blues, png_output=None, show=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title='Normalized confusion matrix'
    else:
        title='Confusion matrix'

    # Calculate chart area size
    leftmargin = 0.5 # inches
    rightmargin = 0.5 # inches
    categorysize = 0.5 # inches
    figwidth = leftmargin + rightmargin + (len(classes) * categorysize)           

    f = plt.figure(figsize=(figwidth, figwidth))

    # Create an axes instance and ajust the subplot size
    ax = f.add_subplot(111)
    ax.set_aspect(1)
    f.subplots_adjust(left=leftmargin/figwidth, right=1-rightmargin/figwidth, top=0.94, bottom=0.1)

    res = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)
    plt.colorbar(res)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if png_output is not None:
        os.makedirs(png_output, exist_ok=True)
        f.savefig(os.path.join(png_output,'confusion_matrix.png'), bbox_inches='tight')

    if show:
        plt.show()
        plt.close(f)
    else:
        plt.close(f)