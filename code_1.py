import cv2
import glob

#put your own folder in the bracket eg. (“myfile/*jpg”)
path = glob.glob("/home/iiwa-2/Downloads/Datasets/ycbv/train_pbr/000000x/rgb/*jpg")
i = 0
for file in path:
    img = cv2.imread(file)
    cv2.imshow("Image", img)
    i = i + 1
    #below is the timing in milisecs on how long you want to show the image
    #putting 0 means you need to manually close the image to see the next image
    print(i)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
