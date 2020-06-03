import imageio


img = imageio.imread('/home/jinen/PycharmProjects/ML-Learn/repo/ch15/example-image.png')
print('Image Shape', img.shape)
print('Number of Channels:', img.shape[2])
print('Image DataType:', img.dtype)
print(img[100:102, 100:102, :])
