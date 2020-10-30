import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


fig = plt.figure()


def f(x, y):
    return np.sin(x) + np.cos(y)

x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
ims = []
for i in range(60):
    x += np.pi / 15.
    y += np.pi / 20.
    im = plt.imshow(f(x, y), animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)

# ani.save('dynamic_images.mp4')

plt.show()




# print(len(output_imgs), cap.shape)
  # figure, axes = plt.subplots(nrows=1, ncols=2)
  # img = None
  # for i in range(cap.shape[0]):
  #   if img is None:
  #     plt.subplot(121)
  #     img = plt.imshow(output_imgs[i])
  #   else:q
  #     img.set_data(output_imgs[i])
  #   # plt.pause(.1)
  #   # plt.draw()

  # img = None
  # plt.subplot(122)
  # for file in cap:
  #   if img is None:
  #     img = plt.imshow(file)
  #   else:
  #     img.set_data(file)
  #   plt.pause(.1)
  #   plt.draw()
  #   # input("<Hit enter to close>")
  #   # plt.close()
  # e()
