import tensorflow as tf
import numpy as np
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