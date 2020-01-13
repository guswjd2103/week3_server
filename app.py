from threading import Thread

from flask import Flask, render_template, Response, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import img_face
import video_face
import cv2
from werkzeug.datastructures import ImmutableMultiDict

UPLOAD_FOLDER = 'uploaded_files'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/hi', methods=['GET'])
def index():
    print("hello")
    return "hihi"


# Image file 받아옴
@app.route('/uploadImage', methods=['POST'])
def upload_image():
    f = request.files['img']
    fname = f.filename
    f.save("./originImage/" + secure_filename(fname))       # 파일 저장

    face_recog = img_face.FaceRecog(fname)
    print(face_recog.known_face_names)
    frame = face_recog.get_frame()
    cv2.imshow("Frame", frame)
    cv2.imwrite('./mosaicImage/' + fname, frame)

    print(request.files)
    print("end")

    return jsonify(fname)

# mosaic image file 보냄
@app.route('/mosaicImage/<filename>', methods=['GET'])
def read_mosaic_img(filename):
    print("파일이름!"+filename)
    print("")
    frame = cv2.imread('./mosaicImage/'+filename)

    cv2.imshow("Frame", frame)

    path = './mosaicImage/'+filename

    return send_file(path, mimetype='image/gif')
    # return jsonify(filename)
    # return Response(frame, mimetype='multipart/x-mixed-replace; boundary=frame')





# Video file 받아옴
@app.route('/uploadVideo', methods=['POST'])
def upload_video():
    f = request.files['img']
    fname = f.filename
    f.save("./originVideo/" + secure_filename(fname))       # 파일 저장

    # thread = Thread(target=video_face.video_face_recog(), args=(fname,) )
    # thread.start()
    video_face.video_face_recog(fname)

    print(request)
    print(request.files)
    print("video end")

    return jsonify(fname)







# Video file 보냄
@app.route('/mosaicVideo/<filename>', methods=['GET'])
def read_mosaic_video(filename):
    print("파일이름!"+filename)
    print("")
    # frame = cv2.imread('./mosaicVideo/'+filename)
    # cv2.imshow("Frame", frame)

    path = './mosaicVideo/'+filename

    return send_file(path, mimetype='video/mp4')
    # return jsonify(filename)
    # return Response(frame, mimetype='multipart/x-mixed-replace; boundary=frame')






if __name__ == '__main__':
    app.run(app.run(host='0.0.0.0', port=80, debug=True))
    # host='127.0.0.1', port=6380, debug=True