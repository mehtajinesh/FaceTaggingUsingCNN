f=open('face_rect.txt','r+')
lines=f.read().split('\n')
result =[]
to_change=[]
for i in lines:
    temp = i.split("\t")
    temp1 = temp[1].split('/')
    print (temp1[2])
    result.append(temp)
    #print to_change #faceId	imagePath	faceRect.x	faceRect.y	faceRect.w	faceRect.h