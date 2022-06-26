from tensorflow.keras.optimizers import Adam
init_lr = 1e-4
epochs = 100
opt = Adam(learning_rate=init_lr, decay=init_lr / epochs)


modelA.compile(optimizer=opt,
              loss = ["mse", # for age
                      "binary_crossentropy"], # for gender
              loss_weights = [4.,  # for age
                              0.1], # for gender
              metrics = ["mae",    # for age
                         "accuracy"]) # for gender

from keras.callbacks import ModelCheckpoint

callbacks = [
     ModelCheckpoint("./model_checkpoint", monitor='val_loss')
]

def generator_wrapper(generator):
  for batch_x,batch_y in generator:
    yield (batch_x,[batch_y[:,i] for i in range(2)])

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

history = modelA.fit_generator(
                    generator=generator_wrapper(train_generator),
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs = 1,
                    verbose= 2,
                    validation_data = generator_wrapper(valid_generator),
                    validation_steps=70)


model_folder = '/content/drive/MyDrive/Colab Notebooks/models'
modelA.save(model_folder+"age_gender_A.h5")
