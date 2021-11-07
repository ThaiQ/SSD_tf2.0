import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.optimizers import SGD, Adam

from ssd_loss import CustomLoss
from utils import bouding_box_util
import getData
from ssd300 import ssd_300

train_data, info = getData.get_dataset("voc/2012", "train")
valid_data, info_val = getData.get_dataset("voc/2012", "validation")
test_data, info_test = getData.get_dataset("voc/2012", "test")
train_total_items = getData.get_total_item_size(info, "train")
val_total_items = getData.get_total_item_size(info_val, "validation")
test_total_items = getData.get_total_item_size(info, "test")
