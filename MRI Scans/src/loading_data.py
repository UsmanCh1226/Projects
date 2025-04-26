import os
import cv2
import glob



def extract_images(directory, img_size=128, extension='.jpg'):
    images_list = []
    ending_extension = '.jpg'

    for filename in glob.glob(os.path.join(directory, "*")):
        if filename.lower().endswith(extension):
            try:
                image = cv2.imread(filename)
                image = cv2.resize((image, (img_size, img_size)))
                images_list.append(image)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return images_list

glioma_directory = 'Users/usmanchaudhry/PycharmProjects/Projects/MRI Scans/Training/glioma'
meningioma_directory = 'Users/usmanchaudhry/PycharmProjects/Projects/MRI Scans/Training/meningioma'
no_tumor_directory = 'Users/usmanchaudhry/PycharmProjects/Projects/MRI Scans/Training/notumor'
pituitary_directory = 'Users/usmanchaudhry/PycharmProjects/Projects/MRI Scans/Training/pituitary'

glioma_images=extract_images(glioma_directory)
meningioma_images=extract_images(meningioma_directory)
no_tumor_images=extract_images(no_tumor_directory)
pituitary_images=extract_images(pituitary_directory)

images_directory={"glioma":glioma_images,"meningioma":meningioma_images,"no tumor":no_tumor_images,"pituitary":pituitary_images}


