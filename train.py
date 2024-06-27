import model
from tensorflow import keras
import numpy as np

myModel = model.createModel(input_img_size = (640, 480, 3))


myModel.compile(
    model_F_optimizer=keras.optimizers.Adam(learning_rate=2e-1, beta_1=0.5),
    model_S_optimizer=keras.optimizers.Adam(learning_rate=2e-1, beta_1=0.5),
    model_F_loss_fn=keras.losses.MeanAbsoluteError(),
    model_S_loss_fn=keras.losses.MeanAbsoluteError()
)

x_train = np.empty([1,640, 480, 3])
y_train = np.ones([1,640, 480, 3])


history = myModel.fit(x_train, y_train, batch_size=1, epochs=10)

print(history.history)


