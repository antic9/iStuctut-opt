import dataset_CNN

struc = "conv77 stride1 padding1 average_pooling22 stride1 Sigmoid fully_connected36 output10"
struc_list = struc.split(" ")
print(struc_list)
dataset_CNN.set_acc(struc_list)
