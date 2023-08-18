import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

im = Image.open(os.path.join('./frames/Saved_frame/fig_vis/33006_sessie1_taskrobotEngagement/00000.jpg'))

# Create figure and axes
fig, ax = plt.subplots()

# Display the image
ax.imshow(im)

# (225, 437), (412, 480), (83, 162), (268, 480), (602, 420), (852, 480)
# Create a Rectangle patch
rect = patches.Rectangle((159, 143), 350-159, 460-143, linewidth=1, edgecolor='r', facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)

plt.show()
