import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Weights initializer for the layers.
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
# Gamma initializer for instance normalization.
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

def downsample_2_dim(
    x,
    filters,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    filters = int(filters)
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = layers.GroupNormalization(groups=1,gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x

def downsample_1_dim(
    x,
    filters,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(7, 1),
    strides=(2, 1),
    padding="same",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    filters = int(filters)
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = layers.Conv2D(
        filters,
        kernel_size[::-1],
        strides=strides[::-1],
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = layers.GroupNormalization(groups=1,gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x
    
def samesample_2_dim(
    x,
    filters,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="same",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    filters = int(filters)
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    
    x = layers.GroupNormalization(groups=1,gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x
    
def samesample_1_dim(
    x,
    filters,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(7, 1),
    strides=(1, 1),
    padding="same",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    filters = int(filters)
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    
    x = layers.Conv2D(
        filters,
        kernel_size[::-1],
        strides=strides[::-1],
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = layers.GroupNormalization(groups=1,gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x

def upsample(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    kernel_initializer=kernel_init,
    gamma_initializer=gamma_init,
    use_bias=False,
):
    filters = int(filters)
    x = layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias,
    )(x)
    x = layers.GroupNormalization(groups=1,gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x
    
def createFullModel(
    input_img_size = (640, 480, 3),
    filters=64
):
    img_input = layers.Input(shape=input_img_size)
    
    x_01 = downsample_1_dim(img_input, filters=filters, kernel_size=(11, 1) ,activation=layers.Activation("relu"))
    x_02 = downsample_1_dim(img_input, filters=filters/2, kernel_size=(7, 1) ,activation=layers.Activation("relu"))
    x_03 = downsample_1_dim(img_input, filters=filters/4, kernel_size=(5, 1) ,activation=layers.Activation("relu"))
    
    x_04 = layers.Concatenate()([x_01, x_02, x_03])
    
    x_05 = samesample_2_dim(x_04, filters=filters, kernel_size=(1, 1) ,activation=layers.Activation("relu"))
    
    x_11 = downsample_1_dim(x_05, filters=filters, kernel_size=(7, 1) ,activation=layers.Activation("relu"))
    x_12 = downsample_1_dim(x_05, filters=filters/2, kernel_size=(5, 1) ,activation=layers.Activation("relu"))
    x_13 = downsample_2_dim(x_05, filters=filters/4, kernel_size=(3, 3) ,activation=layers.Activation("relu"))
    
    x_14 = layers.Concatenate()([x_11, x_12, x_13])
    
    x_15 = samesample_2_dim(x_14, filters=filters, kernel_size=(1, 1) ,activation=layers.Activation("relu"))
    
    x_20 = samesample_1_dim(x_15, filters=filters/2, kernel_size=(5, 1) ,activation=layers.Activation("relu")) 
    x_21 = samesample_2_dim(x_15, filters=filters/2, kernel_size=(3, 3) ,activation=layers.Activation("relu"))
    x_22 = samesample_2_dim(x_21, filters=filters/2, kernel_size=(3, 3) ,activation=layers.Activation("relu"))
    
    x_23 = layers.Concatenate()([x_20, x_22])

    #independent
    x_24 = layers.Conv2D(
        filters,
        (3, 3),
        strides=(1, 1),
        kernel_initializer=kernel_init,
        padding="same",
        use_bias=False,
    )(x_23)
    
    x_24 = layers.GroupNormalization(groups=1,gamma_initializer=gamma_init, name="sub_layer")(x_24)
    x_24 = layers.Activation("relu")(x_24)
    #independent
        
    x_30 = upsample(x_24, filters=filters, kernel_size=(3, 3) ,activation=layers.Activation("relu"))
    x_31 = samesample_1_dim(x_05, filters=filters/2, kernel_size=(7, 1) ,activation=layers.Activation("relu"))
    x_32 = samesample_1_dim(x_05, filters=filters/2, kernel_size=(5, 1) ,activation=layers.Activation("relu"))
    x_33 = samesample_2_dim(x_05, filters=filters/4, kernel_size=(3, 3) ,activation=layers.Activation("relu"))
    
    x_34 = layers.Concatenate()([x_30, x_31, x_32, x_33])
    
    x_35 = samesample_2_dim(x_34, filters=filters, kernel_size=(1, 1) ,activation=layers.Activation("relu"))
    
    x_36 = samesample_1_dim(x_35, filters=filters/2, kernel_size=(7, 1) ,activation=layers.Activation("relu"))
    x_37 = samesample_1_dim(x_35, filters=filters/4, kernel_size=(5, 1) ,activation=layers.Activation("relu"))
    x_38 = samesample_2_dim(x_35, filters=filters/4, kernel_size=(3, 3) ,activation=layers.Activation("relu"))
    
    x_39 = layers.Concatenate()([x_36, x_37, x_38])
    
    x_40 = upsample(x_39, filters=filters, kernel_size=(3, 3) ,activation=layers.Activation("relu"))
    
    x_41 = samesample_1_dim(x_40, filters=filters/2, kernel_size=(11, 1) ,activation=layers.Activation("relu"))
    x_42 = samesample_1_dim(x_40, filters=filters/4, kernel_size=(7, 1) ,activation=layers.Activation("relu"))
    x_43 = samesample_1_dim(x_40, filters=filters/4, kernel_size=(5, 1) ,activation=layers.Activation("relu"))
    x_44 = samesample_2_dim(x_40, filters=filters/4, kernel_size=(3, 3) ,activation=layers.Activation("relu"))
    
    x_45 = layers.Concatenate()([x_41, x_42, x_43, x_44])
    
    x_46 = samesample_2_dim(x_45, filters=filters, kernel_size=(1, 1) ,activation=layers.Activation("relu"))
    
    x_47 = samesample_1_dim(x_46, filters=filters/2, kernel_size=(11, 1) ,activation=layers.Activation("relu"))
    x_48 = samesample_1_dim(x_47, filters=filters/4, kernel_size=(7, 1) ,activation=layers.Activation("relu"))
    x_49 = samesample_1_dim(x_48, filters=filters/8, kernel_size=(5, 1) ,activation=layers.Activation("relu"))
    x_50 = samesample_2_dim(x_49, filters=3, kernel_size=(1, 1) ,activation=layers.Activation("sigmoid"))
    

    model = keras.models.Model(img_input, x_50)
    return model


class MyModel(keras.Model):
    def __init__(
        self,
        full_model,
        sub_model,
    ):
        super().__init__()
        self.model_F = full_model
        self.model_S = sub_model
    
    def compile(
        self,
        model_F_optimizer,
        model_S_optimizer,
        model_F_loss_fn,
        model_S_loss_fn,
    ):
        super().compile()
        self.model_F_optimizer = model_F_optimizer
        self.model_S_optimizer = model_S_optimizer
        self.model_F_loss_fn= model_F_loss_fn
        self.model_S_loss_fn= model_S_loss_fn


    def train_step(self, batch_data):
        real_x, real_y = batch_data


        with tf.GradientTape(persistent=True) as tape:
            f_y = self.model_F(real_x)
            s_y = self.model_S(real_x)

            samef_y = self.model_F(real_y)
            sames_y = self.model_S(real_y)

            f_loss = self.model_F_loss_fn(real_y, f_y)
            s_loss = self.model_S_loss_fn(real_y[: ,::4, ::4, :], s_y)

            samef_loss = self.model_F_loss_fn(real_y, samef_y)
            sames_loss = self.model_S_loss_fn(real_y[: ,::4, ::4, :], sames_y)
            
            allf_loss = (f_loss + samef_loss) * 0.5
            alls_loss = (s_loss + sames_loss) * 0.5

        f_grads = tape.gradient(allf_loss, self.model_F.trainable_variables)
        s_grads = tape.gradient(alls_loss, self.model_S.trainable_variables)

        self.model_F_optimizer.apply_gradients(
            zip(f_grads, self.model_F.trainable_variables)
        )

        self.model_S_optimizer.apply_gradients(
            zip(s_grads, self.model_S.trainable_variables)
        )

        return {
            "F_loss": f_loss,
            "S_loss": s_loss
        }

def createModel(input_img_size = (640, 480, 3)):
    fullmodel = createFullModel()
    sub_model = None

    sub_layer = None
    for layer in fullmodel.layers:
        if(layer.name == "sub_layer"):
            sub_layer = layer.output
            break


    if not (sub_layer is None):
        sub_layer = samesample_2_dim(sub_layer, filters=3, kernel_size=(1, 1) ,activation=layers.Activation("sigmoid"))
        sub_model = keras.models.Model(fullmodel.input, sub_layer)

    sub_model.summary()
    myModel = MyModel(
        full_model = fullmodel, sub_model = sub_model
    )


    return myModel
        


