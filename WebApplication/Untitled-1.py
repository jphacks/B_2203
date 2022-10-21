import face_recognition # Dlib library

dlib_img = face_recognition.load_image_file("C:/Users/Masaharu/Downloads/IMG_9688.jpg")
print(face_recognition.face_locations(dlib_img)) #hog+svm
print(face_recognition.face_locations(dlib_img, model="cnn")) #CNN
print(type(dlib_img))