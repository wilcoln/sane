from icecream import ic

from src.datasets.esnli import get_dataset

# train_set = get_dataset('train')
# val_set = get_dataset('val')
test_set = get_dataset('test')

ic(train_set[:5])