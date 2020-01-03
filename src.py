from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
import numpy as np
from ClothesRecognizer import ClothesRecognizer

clothes_recognizer = ClothesRecognizer()

i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
clothes_recognizer.plot_image(i)
plt.subplot(1, 2, 2)
clothes_recognizer.plot_value_array(i)
plt.show()

img = clothes_recognizer.test_images[1]
img = (np.expand_dims(img, 0))
predictions_single = clothes_recognizer.model.predict(img)


clothes_recognizer.plot_value_array(1)
_ = plt.xticks(range(10), clothes_recognizer.class_names, rotation=45)
clothes_recognizer.plot_image(1)

print(predictions_single)
print(img.shape)
