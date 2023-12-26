import os
import cv2

path= input("Enter Path : \n")
#C:\\Users\\vaibh\\OneDrive\\Desktop\\vaibhav project\\SOCOFing\\Altered\\Altered-Easy\\1__M_Left_index_finger_CR.BMP
sample = cv2.imread(path)

counter = best_score = 0
filename = image = kp1 = kp2 = mp = None
for file in os.listdir(r"C:\Users\vaibh\OneDrive\Desktop\vaibhav project\SOCOFing\Real"):
    if counter % 10 == 0:
        #print(counter)
        print("Searching through image :---> " +  file )
    counter += 1
    fingerprint_image = cv2.imread("C:\\Users\\vaibh\\OneDrive\\Desktop\\vaibhav project\\SOCOFing\\Real\\"+ file)
    sift = cv2.SIFT_create()
    keypoint_1, descriptor_1 = sift.detectAndCompute(sample, None)
    keypoint_2, descriptor_2 = sift.detectAndCompute(fingerprint_image, None)

    matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {}).knnMatch(descriptor_1, descriptor_2, k=2)


    match_points = []

    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            match_points.append(p)

    keypoints = 0
    if len(keypoint_1) < len(keypoint_2):
        keypoints = len(keypoint_1)
    else:
        keypoints = len(keypoint_2)

    if len(match_points) / keypoints * 100 > best_score:
        best_score = len(match_points) / keypoints * 100
        filename = file
        image = fingerprint_image
        kp1, kp2, mp = keypoint_1, keypoint_2, match_points

print("Percentage Match : " + str(best_score))

result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
result = cv2.resize(result, None, fx=3, fy=3)
cv2.imshow("RESULT", result)
cv2.waitKey(0)
cv2.destroyAllWindows()