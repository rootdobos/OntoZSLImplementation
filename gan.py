#https://www.tensorflow.org/tutorials/generative/dcgan
#https://keras.io/examples/generative/dcgan_overriding_train_step/
import tensorflow as tf
import time

def make_generator_model():
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(4096, use_bias=False,input_shape=(200,)))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(2048, use_bias=False,input_shape=(200,)))
    return model

def make_discriminator_model():
    model= tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(2048, input_shape=(2048,)))
    model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.Dense(1))
    return model

def discriminator_loss(real_output,fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss=cross_entropy(tf.ones_like(real_output),real_output)
    fake_loss=cross_entropy(tf.zeros_like(fake_output),fake_output)
    total_loss= real_loss+fake_loss
    return total_loss

def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output),fake_output)
def generator_loss_classifier(prediction, ground_truth):
    sc_cross_entropy= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    return sc_cross_entropy(ground_truth,prediction)
@tf.function
#def train_step(features, seeds_from_ontology, generator,discriminator,generator_optimizer,discriminator_optimizer):
def train_step( seeds_from_ontology, generator,classifier,ground_truth,generator_optimizer):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_features=generator(seeds_from_ontology,training=True)

        #real_output= discriminator(features, training=True)
        #fake_output= discriminator(generated_features,training=True)
        prediction =classifier(generated_features,training=False)
        # print(prediction.shape)
        # print(ground_truth.shape)
        #gen_loss= generator_loss(fake_output)
        #disc_loss= discriminator_loss(real_output,fake_output)
        gen_loss= generator_loss_classifier(prediction,ground_truth)
    gradients_of_generator= gen_tape.gradient(gen_loss,generator.trainable_variables)
    #gradients_of_discriminator= disc_tape.gradient(disc_loss,discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    #discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    #return gen_loss,disc_loss
    return gen_loss

#def train(features, seeds,epochs, generator, discriminator, generator_optimizer,discriminator_optimizer,checkpoint,checkpoint_prefix):
def train(seeds,epochs, generator, classifier,ground_truth, generator_optimizer,checkpoint,checkpoint_prefix):
    for epoch in range(epochs):
        start=time.time()
        for i in range(seeds.shape[0]):
            #gen_loss, disc_loss=train_step(features[i],seeds[i],generator,discriminator,generator_optimizer,discriminator_optimizer)
            gen_loss=train_step(seeds[i],generator,classifier,ground_truth[i],generator_optimizer)
        if(epoch+1)%15==0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        print('Generator loss: {} '.format(gen_loss))