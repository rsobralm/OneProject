import os
import random
import shutil


train_path = 'data/train/'
test_path = 'data/test/'

for dir in os.listdir(train_path):
    os.makedirs(test_path + dir,  exist_ok=True)
    src_path = train_path + dir
    dst_path = test_path + dir

    list_instances = os.listdir(src_path)

    test_set_size = int(len(list_instances)/4) + 1

    for i in range(test_set_size):
        selected_file = random.choice(list_instances)
        shutil.move(src_path+'/'+selected_file, dst_path+'/'+selected_file)
        list_instances.remove(selected_file)
    

