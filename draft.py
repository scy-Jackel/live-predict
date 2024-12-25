# save_num = 0
#
# for npy in npy_list:
#     npy_max = npy.max()
#     npy = npy/float(npy_max)*255
#
#     depth = npy.shape[0]
#     out_path = './view'+str(save_num)+'/'
#     if os.path.exists(out_path):
#         shutil.rmtree(out_path)
#     if not os.path.exists(out_path):
#         os.makedirs(out_path)
#     for k in range(depth):
#         slice0 = npy[k]
#         cv2.imwrite(out_path+'a_'+str(k)+'.jpg',slice0)
#
#
#
#     save_num+=1