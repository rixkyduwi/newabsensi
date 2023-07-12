from django.shortcuts import render,redirect
from django.http import StreamingHttpResponse
from django.contrib.auth import authenticate,login,logout
from django.contrib import messages
from .forms import CreateUserForm
from django.contrib.auth.decorators import login_required
from mysite.settings import BASE_DIR
from .models import student_profile,student_attendance
from django.core.mail import EmailMessage
from django.conf import settings
from .decorators import allowed_users
import os
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import time
import datetime
import csv



@login_required(login_url='login')
@allowed_users(allowed_roles=['teacher','admin'])
def create_dataset(request):        
    Id=request.POST['userId']
    name=request.POST['userId1']
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(BASE_DIR+'render/algorithms/haarcascade_frontalface_default.xml')
    sampleNum=0
    while(True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
            #incrementing sample number 
            sampleNum=sampleNum+1
            #saving the captured face in the dataset folder TrainingImage
            cv2.imwrite(BASE_DIR+"/static/images/TrainingImage/"+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
            #display the frame
            cv2.imshow('frame',img)
            #wait for 100 miliseconds 
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        # break if the sample number is morethan 100
        elif sampleNum>60:
            break
    cam.release()
    cv2.destroyAllWindows() 
    # res = "Images Saved for ID : " + Id +" Name : "+ name
    row = [Id , name]
    with open('render/StudentDetails/StudentDetails.csv','a+') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()
    return redirect('home')

@allowed_users(allowed_roles=['teacher','admin'])
@login_required(login_url='login')
def trainer(request):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(BASE_DIR+'render/algorithms/haarcascade_frontalface_default.xml')
    faces,Id = getImagesAndLabels(BASE_DIR+"/static/images/TrainingImage/")
    recognizer.train(faces, np.array(Id))
    recognizer.save(BASE_DIR+"render/algorithms/TrainingImageLabel/Trainner.yml")
    cv2.destroyAllWindows()
    return redirect('home')

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #print(imagePaths)
    
    #create empth face list
    faces=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)     
        cv2.imshow("Training", imageNp)
        cv2.waitKey(10)   
    return faces,Ids

def video_feed(request):
    return StreamingHttpResponse(get_video_stream(request), content_type='multipart/x-mixed-replace; boundary=frame')


def get_video_stream(request):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(BASE_DIR+'render/algorithms/TrainingImageLabel/Trainner.yml')
    harcascadePath = "render/algorithms/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)   
    df = pd.read_csv(BASE_DIR + "render/StudentDetails/StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)   
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)
    saved_data = False  # Flag untuk menandakan apakah data sudah disimpan atau belum

    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x+w, y+h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            
            if conf < 100:
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = df.loc[df['Id'] == Id]['Name'].values[0]
                name = aa  # Menggunakan aa langsung sebagai nilai name tanpa metode join()
                tt = str(int(Id)) + "-" + aa
                
                if not saved_data:  # Simpan data hanya jika belum disimpan sebelumnya
                    attendance.loc[len(attendance)] = [Id, name, date, timeStamp]
                    student = student_profile.objects.get(student_id=Id)
                    data = student_attendance.objects.create(roll=student, name=name, date=date, time=timeStamp)
                    data.save()
                    user = student_profile.objects.filter(student_id=Id).update(attendance='Present')
                    saved_data = True  # Set flag ke True setelah data disimpan
            else:
                Id = 'Unknown'                
                tt = str(Id)  
            
            if conf > 75:
                noOfFile = len(BASE_DIR + ("ImagesUnknown")) + 1
                cv2.imwrite("ImagesUnknown/Image" + str(noOfFile) + ".jpg", im[y:y+h, x:x+w])            
            cv2.putText(im, str(tt), (x, y+h), font, 1, (255, 255, 255), 2)        
        
        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')    
        ret, frame = cv2.imencode('.jpg', im)
        frame = frame.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame +
               b'\r\n')
        
        

def TrackImages(request):
    return redirect(request, 'dashboard.html')
    

def addStudent(request):
    if request.method=='POST':
        id = request.POST.get('rollno', '')
        name = request.POST.get('username','')
        address=request.POST.get('address', '')
        mob= request.POST.get('mobileno', '')
        email=request.POST.get('email', '')
        desc=request.POST.get('desc','')
        user=student_profile(student_id=id,name=name,address=address,phone=mob,email=email,description=desc)
        user.save()
        if user:
            messages.success(request, 'Account was created sfor ' + name)
            return redirect('home')
        else:
            return messages.success(request,'Internal Server Error')
    return render(request, 'render/add_student.html')

def registerPage(request):
    form=CreateUserForm()
    if request.method=='POST':
        form=CreateUserForm(request.POST)
        if form.is_valid():
            user=form.save()
            username=form.cleaned_data.get('username')
            messages.success(request,'Account Was Created For '+username)
            return redirect('login')
    context={'form':form}
    return render(request,'render/add_teacher.html',context)

def loginPage(request):
    if request.method=='POST':
        username=request.POST.get('username','')
        password=request.POST.get('password','')
        user=authenticate(request,username=username,password=password)
        if user is not None and user.is_staff:
            login(request,user)
            return redirect('home')
        elif user is not None and user.is_active:
            login(request, user)
            return redirect('student')
        else:
            messages.info(request,'Username Or Password is Incorrect')
    context={}
    return render(request,'render/login.html',context)

@login_required(login_url='login')
@allowed_users(allowed_roles=['teacher','admin'])
def deleteStudent(request,pk):
    student=student_profile.objects.get(student_id=pk)
    if request.method=='POST':
        student.delete()
        return redirect('/')
    context={'student':student}
    return render(request,'render/delete.html',context)

def index(request):
    return render(request,'studentpage/index.html')

@login_required(login_url='login')
def home(request):
    if request.user.is_staff:
        return render(request,'render/dashboard.html')
    else:
        return render(request, 'studentpage/dashboard.html')

def logoutUser(request):
    logout(request)
    return redirect('login')

@login_required(login_url='login')
@allowed_users(allowed_roles=['teacher','admin','user'])
def profile(request,pk):
    id = int(pk)
    request.FILES
    if request.method=='POST':
        id = int(pk)
        name = request.POST.get('username','')
        address=request.POST.get('address', '')
        mob= request.POST.get('mobileno', '')
        email=request.POST.get('email', '')
        desc=request.POST.get('desc','')
        user=student_profile(student_id=id,name=name,address=address,phone=mob,email=email,description=desc)
        user.save()
        if user:
            messages.success(request, 'Account was Updated For ' + name)
        else:
            return messages.success(request,'Internal Server Error')
        render(request,'render/student_profile.html')
        
    student=student_profile.objects.filter(student_id=id)
    attendance=student_attendance.objects.filter(roll=id)
    present=student_attendance.objects.filter(roll=pk).count()
    absent=60-present
    context={'student':student,'attendance':attendance,'present':present,'absent':absent}
    return render(request,'render/student_profile.html',context)

@login_required(login_url='login')
# @allowed_users(allowed_roles=['teacher','admin'])
def all_students(request):
    student=student_profile.objects.all()
    total_students_absent=student_profile.objects.filter(attendance='Absent').count()
    total_students_present=student_profile.objects.filter(attendance='Present').count()
    total_students=student_profile.objects.count()
    context={'student':student,'total_students':total_students,'total_students_present':total_students_present,'total_students_absent':total_students_absent}
    return render(request,'render/total.html',context)
    
def about(request):
    return render(request,'render/about.html')
   
@login_required(login_url='login')
# @allowed_users(allowed_roles=['teacher','admin'])
def absent_students(request):
    student=student_profile.objects.filter(attendance='Absent').all()
    total_students_absent=student_profile.objects.filter(attendance='Absent').count()
    total_students_present=student_profile.objects.filter(attendance='Present').count()
    total_students=student_profile.objects.count()
    context={'student':student,'total_students':total_students,'total_students_present':total_students_present,'total_students_absent':total_students_absent}
    
    return render(request,'render/total.html',context)
    
@login_required(login_url='login')
@allowed_users(allowed_roles=['teacher','admin'])
def report(request):
    
    students = student_profile.objects.all()
    attendance_data = student_attendance.objects.all()
    total_students_present = student_profile.objects.filter(attendance='Present').count()
    total_students = student_profile.objects.count()
    ts = time.time()  
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%M-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    selected_date = request.GET.get('date')  # Retrieve the selected date from the query parameters
    
    filtered_data = student_attendance.objects.filter(date=selected_date)  # Filter the data based on the selected date
    # Combine attendance data with student data
    combined_data = []
    for attendance in attendance_data:
        for student in students:
            if attendance.roll_id == student.student_id:
                combined_data.append({
                    'roll': attendance.roll_id,
                    'name': student.name,
                    'date': attendance.date,
                    'time': attendance.time,
                    'attendance': student.attendance
                })
    
    context = {
        'data': combined_data,
        'total_students': total_students,
        'total_students_present': total_students_present,
        'date': date,
        'time': timeStamp,
        'filtered_data': filtered_data
    }
    
    return render(request, 'render/report.html', context)

@login_required(login_url='login')
def present(request):
    students = student_profile.objects.all()
    attendance_data = student_attendance.objects.all()
    total_students_present = student_profile.objects.filter(attendance='Present').count()
    total_students = student_profile.objects.count()
    ts = time.time()  
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')  # Mengubah format tanggal
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    
    filtered_data = student_attendance.objects.filter(date=date)  # Filter data berdasarkan tanggal hari ini
    # Menggabungkan data kehadiran dengan data mahasiswa
    combined_data = []
    for attendance in attendance_data:
        for student in students:
            if attendance.roll_id == student.student_id:
                combined_data.append({
                    'roll': attendance.roll_id,
                    'name': student.name,
                    'date': attendance.date,
                    'time': attendance.time,
                    'attendance': student.attendance
                })
    
    context = {
        'data': combined_data,
        'total_students': total_students,
        'total_students_present': total_students_present,
        'date': date,
        'time': timeStamp,
        'filtered_data': filtered_data
    }
    
    return render(request,'studentpage/presensi.html',context)
  

@login_required(login_url='login')
@allowed_users(allowed_roles=['teacher','admin'])
def send_file(request):
    return render(request,'render/file.html')

@login_required(login_url='login')
@allowed_users(allowed_roles=['teacher','admin'])
def send(request):
    if request.method=='POST':
        request.POST, request.FILES
        subject = request.POST.get('title')
        message = request.POST.get('subject')
        email = request.POST.get('email')
        files = request.FILES.getlist('file')
        try:
            mail = EmailMessage(subject, message, settings.EMAIL_HOST_USER, [email])
            for f in files:
                mail.attach(f.name, f.read(), f.content_type)
            mail.send()
            # messages.success(request,"File Sent To " + email)
            return render(request,'render/file.html',{'error_message': 'Sent email to %s'%email})
        except:
            return render(request,'render/file.html',{'error_message': 'Either the attachment is too big or corrupt'})

    return render(request,'render/file.html',{'error_message': 'Unable to send email. Please try again later'})