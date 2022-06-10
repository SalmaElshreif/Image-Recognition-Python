from keras.applications.vgg16 import decode_predictions, VGG16
from keras.applications.vgg16 import preprocess_input

# load the model
from keras.utils import load_img, img_to_array

model = VGG16()
# load an image from file
image = load_img('mug.jpg', target_size=(224, 224))
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
# predict the probability across all output classes
output = model.predict(image)
# convert the probabilities to class labels
label = decode_predictions(output)
# retrieve the most likely result, e.g. highest probability
label1 = label[0][0]
label2 = label[0][1]
label3 = label[0][2]
label4 = label[0][3]


# print the classification
print('%s (%.2f%%)' % (label1[1], label1[2]*100))
print('%s (%.2f%%)' % (label2[1], label2[2]*100))
print('%s (%.2f%%)' % (label3[1], label3[2]*100))
print('%s (%.2f%%)' % (label4[1], label4[2]*100))