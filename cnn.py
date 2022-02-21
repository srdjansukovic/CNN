from keras import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.models import load_model

from prepare_images import prepare, prepare_single_image

classifier = Sequential()

classifier.add(Conv2D(input_shape=(28, 28, 1), activation='relu', filters=32, kernel_size=(3, 3)))
classifier.add(MaxPool2D(pool_size=(2, 2)))
classifier.add(Conv2D(activation='relu', filters=32, kernel_size=(3, 3)))
classifier.add(MaxPool2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=10, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

train_images, train_labels, test_images, test_labels = prepare(root_directory='digits-smaller')
print(train_images.shape)
print(train_labels.shape)


# train or load model

# classifier.fit(train_images, train_labels, batch_size=32, epochs=300)
# classifier.save('trained_model.h5')
classifier = load_model('trained_model.h5')


#   if wanted to feed only one image to the network
# test_image = prepare_single_image('digits-smaller/eval/0/2915.jpg')
# images = [test_image]
# prediction = classifier.predict(np.asarray(images))
# print(prediction)

prediction = classifier.predict(test_images)

correct_guesses = 0

with open('output.txt', 'w') as file_handler:
    for item, label in zip(prediction, test_labels):
        winner = list(item).index(max(item))
        file_handler.write("{0} --> {1}\n".format(winner, label))
        if winner == label:
            correct_guesses += 1

overall_percentage = correct_guesses / len(test_labels)
print('Score: ', overall_percentage)




