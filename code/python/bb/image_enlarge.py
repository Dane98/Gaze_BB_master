import os
from PIL import Image


def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


if __name__ == '__main__':

    image_root = "/home/linlincheng/Gaze_estimation/gaze-automated-annotation/bb/frames/"
    image_in = "frame_invis/"
    image_folder = image_root + image_in

    image_out = "enlarged_frame/frame_invis/"
    saved_folder = image_root + image_out

    images = []
    for img in os.listdir(image_folder):
        if img.endswith(".png"):
            images.append(img)

            im = Image.open(image_folder + img)

            if im.size != (852, 480):
                im = im.resize((852, 480), Image.LANCZOS)
                width, height = im.size
                # im.show()
                im_new = add_margin(im, 0, round(width / 8), round(height * 0.75), round(width / 8), (0, 0, 0))
                im_new.save(saved_folder + img, quality=95)

            width, height = im.size
            im_new = add_margin(im, 0, round(width / 8), round(height * 0.75), round(width / 8), (0, 0, 0))
            im_new.save(saved_folder + img, quality=95)
