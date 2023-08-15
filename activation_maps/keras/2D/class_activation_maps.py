from keras.applications.vgg16 import VGG16
from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import pandas as pd
from keras.applications.vgg16 import decode_predictions
import numpy as np
import seaborn as sns
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
K.clear_session()

model = VGG16(weights='imagenet')

img_path = r'images/sheep.png'
img=mpimg.imread(img_path)
plt.imshow(img)


img = image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)


x = preprocess_input(x)

preds = model.predict(x)
predictions = pd.DataFrame(decode_predictions(preds, top=3)[0], columns=['column', 'categories', 'probability']).iloc[:, 1:]
print('Prediction:', predictions.loc[0, 'categories'])

f = sns.barplot(x='probability', y='categories', data=predictions, color='purple')
sns.set_style(style='white')
f.grid(False)
f.spines['top'].set_visible(False)
f.spines['right'].set_visible(False)
f.spines['bottom'].set_visible(False)
f.spines['left'].set_visible(False)
f.set_title('Top 3 Predictions')

argmax = np.argmax(preds[0])
output = model.output[:, argmax]

model.summary()
last_conv_layer = model.get_layer('block5_conv3')
grads = K.gradients(output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()

img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

hif = .8
superimposed_img = heatmap * hif + img

output = 'output_kit_fox.jpeg'
cv2.imwrite(output, superimposed_img)

img=mpimg.imread(output)
plt.imshow(img)
plt.axis('off')
plt.title(predictions.loc[0, 'categories'])
plt.show()