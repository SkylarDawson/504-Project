import cv2
import mediapipe as mp
from segmentation_book1 import *
import cProfile
from time import time
from imutils.video import VideoStream
import imutils

# from picamera import Picamera

# cap = cv2.VideoCapture(0)    
# cap = cv2.VideoCapture(src=0)
cap = VideoStream(src=0).start()
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Create a VideoWriter object
# Get the frame width and height
#frame_width = int(cap.get(3))
#frame_height = int(cap.get(4))
frame_width = 320
frame_height = 200
#vid = cv2.VideoWriter('output2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, (frame_width, frame_height))

def line_intersection(line1, line2):
    x1, y1 = line1[0]
    x2, y2 = line1[1] #xtip1 and ytip1
    x3, y3 = line2[0]
    x4, y4 = line2[1] #xtip1 and ytip1
    
    # Calculate the determinant
    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # Check if the lines are parallel
    if det == 0:
        return -100,-100

    # Calculate the intersection point
    intersection_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
    intersection_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det
    
    # if both pointing up
    if (y2 < y1) and (y4 < y3): #tip is above knuckle
        #if x point is towards eachother
        if intersection_y < min(y2,y4):
            return int(intersection_x), int(intersection_y)

        
    # if one pointing up
    if (y2 < y1) ^ (y4 < y3): #exclusive or for one hand pointing up
        #if x point is towards eachother
        if intersection_y < max(y2,y4) and intersection_y > min(y2,y4):
            return int(intersection_x), int(intersection_y)


    # if both pointing down
    if (y2 > y1) and (y4 > y3): #tip is above knuckle
        #if x point is towards eachother
        if intersection_y > max(y2,y4):
            return int(intersection_x), int(intersection_y)
        
    return -100,-100

while True:
    try:
        image = cap.read()
        assert(image is not None)
        image = imutils.resize(image, width=frame_width, height=frame_height)
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(imageRGB)
    
        # checking whether a hand is detected
        if results.multi_hand_landmarks:
            pt_list = []
            for handLms in results.multi_hand_landmarks: # working with each hand
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
    
                    # id for tip of index finger
                    if id == 8 :
                        pt_list.append((cx, cy))
                        # cv2.circle(image, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
                    #id for knuckle on index finger
                    if id == 7 :
                        pt_list.append((cx, cy))
                        # cv2.circle(image, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
    
                    # mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)
    
            if(len(pt_list)==4):
                x, y = line_intersection([pt_list[0],pt_list[1]], [pt_list[2], pt_list[3]])
                # cv2.line(image, pt_list[1], (x,y), (0, 255, 255), 2, lineType=cv2.LINE_AA)
                # cv2.line(image, pt_list[3], (x,y), (0, 255, 255), 2, lineType=cv2.LINE_AA)
                height, width, _ = image.shape
                if x < width and y < height and x >= 0 and y >= 0:
                    """
                    super_img = segmentation.slic(image, n_segments=5, compactness=10) - 1
                    super_pix = super_img[y, x]
                    super_img[super_img != super_pix] = 0
                    super_img[super_img == super_pix] = 255
                    """
                    out = segment(image, y, x)
                    cv2.imshow("Segmented", out)
                    # cv2.imshow("test", apply_supermap(image, super_img))
                    # cv2.imshow("test", skimage.color.label2rgb(super_img, image))
                    
                    # seg_image_gray = segment(image, x, y)
                    # seg_image_gray = segment(image, y, x)
                    
                    # out = image.copy()        #np.ndarray((480, 640, 3))
                    # out[:,:,0] = seg_image_gray*255
                    # out[:,:,1] = seg_image_gray*255
                    # out[:,:,2] = seg_image_gray*255

                    #print(image.shape)
                    #print(super_img.shape)
                    # print(ima!mage.color.label2rgb(super_img, image))
                    # cv2.imshow("Segmented", cv2.bitwise_and(image, super_img))
    
                    cv2.circle(image, (x, y), 10, (0, 255, 0), cv2.FILLED)
                    
                    #A = skimage.color.label2rgb(super_img, image)*255
                    #A = np.array(A, dtype=np.uint8)
                    #vid.write(A)
                    
                    
                    # vid.write(image)
                    
                    # blank_frame = np.zeros((height, width, 3), dtype="uint8")
                    # frame = skimage.color.rgb2lab(image)
                    # blank_frame = cv2.addWeighted(blank_frame, 0.5, frame, 0.5, 0)
                    # vid.write(blank_frame)

                    # works for writing segmented images
                    # cv2.imwrite('folder\output_'+str(time())+'.png', 255*skimage.color.label2rgb(super_img, image))
    
        cv2.imshow("Output", image)
        # Write the frame to the output video
        cv2.waitKey(1)
    except KeyboardInterrupt:
        print("end program")
        vid.release()
        cv2.destroyAllWindows()
        break

# Release the video capture and  writer objects
#cap.release()



