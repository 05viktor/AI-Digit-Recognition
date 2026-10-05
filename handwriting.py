import sys
import tensorflow as tf

#MNIST handwritimg set
mnist  = tf.keras.datasets.mnist

#Prep data for training
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train /255.0 , x_test/255.0
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
x_train = x_train.reshape(
    x_train.shape[0], x_train.shape[1], x_train.shape[2], 1
)
x_test = x_test.reshape(
    x_test.shape[0], x_test.shape[1], x_test.shape[2], 1
)

#Convolutional neural network
model = tf.keras.models.Sequential([
    #Strat Convolutional, 32 filtre cu 3x3 kernel
    tf.keras.layers.Conv2D(
        32,(3,3),activation="relu", input_shape=(28,28,1)
    ),
    #Max-pooling layer , 2x2
    tf.keras.layers.MaxPooling2D(
        pool_size=(2,2),
    ),
    #Flatten units
    tf.keras.layers.Flatten(),
    #Hidden layers with dropout - in this case i only need one
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5), #Dropout de 50% ca sa evit overfitting si sa nu fiu
                                  #prea dependent de anumite noduri
    #Output layer with output units for the 10 digits
    tf.keras.layers.Dense(10,activation="softmax")
])

#Train nn
model.compile(
    optimizer ="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(x_train, y_train, epochs=10)

#Performance eval
model.evaluate(x_test, y_test, verbose=2)

#Save model to file - if i want
if len(sys.argv) == 2:
    filename = sys.argv[1]
    model.save(filename)
    print(f"Model saved to {filename}")