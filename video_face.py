import cv2
import face_recognition
import numpy as np
import os

def video_face_recog(video_name):

    print(video_name)

    # 비디오 설정
    input_movie = cv2.VideoCapture("./originVideo/"+video_name)
    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
    codec = int(input_movie.get(cv2.CAP_PROP_FOURCC))
    fps = int(input_movie.get(cv2.CAP_PROP_FPS))
    frame_width = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_movie = cv2.VideoWriter('./mosaicVideo/'+video_name, codec, fps, (frame_width,frame_height))

    # knowns 학습
    known_face_encodings = []
    known_faces = []

    dirname = 'knowns'
    files = os.listdir(dirname)
    for filename in files:
        name, ext = os.path.splitext(filename)  # 파일 이름 == 사람 이름
        if ext == '.jpg':
            known_faces.append(name)
            pathname = os.path.join(dirname, filename)
            img = face_recognition.load_image_file(pathname)  # knowns 사진에서
            face_encoding = face_recognition.face_encodings(img)[0]  # 얼굴 영역 알아냄, 68개 얼굴 특징 위치 분석(face landmarks)
            known_face_encodings.append(face_encoding)

    # image = face_recognition.load_image_file("./knowns/bogum.jpg")
    # face_encoding = face_recognition.face_encodings(image)[0]


    # Initialize variables
    face_locations = []
    face_encodings = []
    face_names = []
    frame_number = 0

    while True:
        print("hello "+ video_name)
        # Grab a single frame of video
        ret, frame = input_movie.read()
        frame_number += 1

        print(frame_number)


        # Quit when the input video file ends
        if not ret:
            break

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        
        # frame = cv2.flip(frame, 0)  # 상하 반전
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)    # model="cnn"
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            min_value = min(distances)

            # tolerance: How much distance between faces to consider it a match. Lower is more strict.
            # 0.6 is typical best performance. 0.6 이상이면 다른 사람
            name = "Unknown"

            if min_value < 0.4:
                index = np.argmin(distances)
                name = known_faces[index]
                print(min_value)

            face_names.append(name)

        # Label the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            print(name)
            if name == "Unknown":
                try:
                    mosaic_img = frame[left:bottom, left:right]  # 이미지 자르고
                    print("r-l " + str((right - left) // 30))
                    print("b-t " + str((bottom - top) // 30))

                    mosaic_img = cv2.resize(mosaic_img, ((right - left) // 100, (bottom - top) // 100))  # 지정 배율로 확대/축소
                    mosaic_img = cv2.resize(mosaic_img, (right - left, bottom - top),
                                            interpolation=cv2.INTER_AREA)  # 원래 크기로 resize
                    frame[top:bottom, left:right] = mosaic_img  # 원래 이미지에 붙이기
                except Exception as e:
                    print(str(e))

            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 40), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 2, (0, 255, 255), 1)

        # Write the resulting image to the output video file
        print("Writing frame {} / {}".format(frame_number, length))
        # cv2.imshow(frame)
        output_movie.write(frame)

    # All done!
    input_movie.release()
    output_movie.release()
    cv2.destroyAllWindows()