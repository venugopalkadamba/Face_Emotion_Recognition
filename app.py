from flask import Flask, render_template, Response

from FER_Camera import VideoCamera

app = Flask(__name__)

@app.route("/", methods = ['GET','POST'])
def index():
    return render_template("index.html")

def generate(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n') 

@app.route("/video_feed")
def video_feed():
    return Response(generate(VideoCamera()),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/video")
def video():
    return render_template("video.html")

if __name__=='__main__':
    app.run(debug=True)