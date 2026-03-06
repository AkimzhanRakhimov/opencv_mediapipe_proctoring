from django.shortcuts import render
from django.http import StreamingHttpResponse
from . import proctoring
# Create your views here.
def home(request):
    return render(request,"myapp/home.html")

def video_feed(request):
    return StreamingHttpResponse(
        proctoring.generate_frames(),
        content_type="multipart/x-mixed-replace; boundary=frame"
    )
