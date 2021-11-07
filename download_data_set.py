from utils import data_utils
import tensorflow_datasets as tfds

train_data,info=data_utils.get_dataset("voc/2012", "train")
data_utils.get_dataset("voc/2012", "validation")
data_utils.get_dataset("voc/2012", "test")

fig = tfds.show_examples(train_data,info)