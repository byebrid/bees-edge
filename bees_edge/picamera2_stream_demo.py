"""
Stream live video feed from Pi to any device on the local network. To use, simply
run this script and then go to http://192.168.0.40:8000/index.html in your
browser, and you should see the video in the centre of the page.

To stop the server, just Ctrl-C to interrupt this script.

Adapted from code presented in this article: https://randomnerdtutorials.com/video-streaming-with-raspberry-pi-camera/
Thanks random nerd!

Needed to be adapted for picamera2, which has a slightly different interface
than picamera. See the picamera2 manual here: https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf.
"""

from picamera2.encoders import H264Encoder, MJPEGEncoder
from picamera2 import Picamera2
from picamera2.outputs import CircularOutput, FileOutput, FfmpegOutput, Output
import time
import io
import logging
import socketserver
from threading import Condition
from http import server


PAGE="""\
<html>
<head>
<title>Raspberry Pi - Camera feed</title>
</head>
<body>
<center><h1>Raspberry Pi - Camera feed</h1></center>
<center><img src="stream.mjpg" width="640" height="480"></center>
</body>
</html>
"""


class StreamingOutput(io.BufferedIOBase):
    def __init__(self, condition):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = condition

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame, copy the existing buffer's content and notify all
            # clients it's available
            self.buffer.truncate()
            with self.condition:
                self.frame = self.buffer.getvalue()
                self.condition.notify_all()
            self.buffer.seek(0)
        return self.buffer.write(buf)

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with condition:
                        condition.wait()
                        frame = stream.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    picam2 = Picamera2()
    video_config = picam2.create_video_configuration()
    picam2.configure(video_config)
    encoder = MJPEGEncoder(bitrate=10000000)

    # This condition is used so we know when exactly a full frame has been
    # written to the buffer
    condition = Condition()
    # `stream` is where we actually retrieve the frames from, it's a wrapper
    # for a BytesIO
    stream = StreamingOutput(condition=condition)
    output = FileOutput(file=stream)

    picam2.start_recording(encoder, output)
    try:
        address = ('', 8000)
        server = StreamingServer(address, StreamingHandler)
        server.serve_forever()
    finally:
        picam2.stop_recording()