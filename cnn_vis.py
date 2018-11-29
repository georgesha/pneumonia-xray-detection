# Visualization CNN
from keras.utils import plot_model
from keras.models import model_from_json

# load model
file = open('cnn3.json', 'r')
cnn_json = file.read()
file.close()
cnn = model_from_json(cnn_json)
# load weights
cnn.load_weights("cnn3.h5")
print("Loaded model")

# plot_model(cnn, to_file='cnn_new.png', show_shapes=True, show_layer_names=True, expand_nested=True)

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

dot = model_to_dot(cnn).create(prog='dot', format='svg')
svg.save("test.svg")
