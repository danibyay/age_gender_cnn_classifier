from tensorflow.keras.optimizers import Adam
init_lr = 1e-4
epochs = 100
opt = Adam(lr=init_lr, decay=init_lr / epochs)


modelA.compile(optimizer=opt,
              loss = ["mse", # for age
                      "binary_crossentropy"], # for gender
              loss_weights = [4.,  # for age
                              0.1], # for gender
              metrics = ["mae",    # for age
                         "accuracy"]) # for gender
