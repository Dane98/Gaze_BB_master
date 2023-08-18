import cv2
import os

folder = '/home/linlincheng/Documents/Projects/L2CS-Net-main/heatmap/plots/simple_mytest6/5s'  # path that saves plots
folder = 'E:/Master/MasterThesis/L2CS-Net-main/heatmap/plots/simple_mytest6/5s/'  # path that saves plots
# images = []
count = 0

for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder, filename))
    if img is not None and filename.endswith(".jpg"):
        # images.append(img)
        cv2.imshow("images", img)
        # the timing of showing the image in milisecs; putting 0 means you need to close the image manually to proceed
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    else:
        count += 1
        print(f"There are {count} images that are not counted.")

# print(len(images))
