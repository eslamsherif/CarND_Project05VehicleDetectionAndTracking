InputFile = "Feature_Extraction_Tuning.csv"

FileHandler = open(InputFile, "r")
lines = FileHandler.readlines()

def nextField():
    global oldindex
    global newindex
    oldindex = newindex + oldindex + 1
    newindex = line[oldindex:].find(",")

def returnBool(x):
    if x == "ON":
        return "True "
    else:
        return "False"

def returnColorSpace(x):
    if x.isdigit() == True:
        return int(x)
    else:
        return "\""+x+"\""

def returnColorSpaceChannelIndex(x):
    if x.isdigit() == True:
        return int(x)
    else:
        return "\""+x+"\""

counter = 0

for line in lines:
    newindex = 0
    oldindex = line.find(",")
    # only data lines start with index
    if line[0:oldindex].isdigit() == True:
        nextField()
        Normalize      = returnBool(line[oldindex : newindex+oldindex])
        nextField()
        CHist_EN       = returnBool(line[oldindex : newindex+oldindex])
        nextField()
        CHist_BCnt     = int(line[oldindex : newindex+oldindex])
        nextField()
        CHist_CSpace   = returnColorSpace(line[oldindex : newindex+oldindex])
        nextField()
        CHist_Ch_Idx   = returnColorSpaceChannelIndex(line[oldindex : newindex+oldindex])
        nextField()
        SB_EN          = returnBool(line[oldindex : newindex+oldindex])
        nextField()
        SB_Size        = "("+line[oldindex : newindex+oldindex]+","+line[oldindex : newindex+oldindex]+")"
        nextField()
        SB_CSpace      = returnColorSpace(line[oldindex : newindex+oldindex])
        nextField()
        HOG_EN         = returnBool(line[oldindex : newindex+oldindex])
        nextField()
        HOG_OrientCnt  = int(line[oldindex : newindex+oldindex])
        nextField()
        HOG_Px_Per_Cl  = int(line[oldindex : newindex+oldindex])
        nextField()
        HOG_Cl_Per_Blk = int(line[oldindex : newindex+oldindex])
        nextField()
        HOG_Blk_Nrmlz  = line[oldindex : newindex+oldindex]
        nextField()
        HOG_Trnsf_Sqrt = returnBool(line[oldindex : newindex+oldindex])
        nextField()
        HOG_CSpace     = returnColorSpace(line[oldindex : newindex+oldindex])
        nextField()
        HOG_Ch_Indx    = returnColorSpaceChannelIndex(line[oldindex : newindex+oldindex])
        nextField()
        
        print("["+Normalize+","+CHist_EN+",["+str(CHist_BCnt)+"],"+CHist_CSpace+","+str(CHist_Ch_Idx)+","+SB_EN+","+str(SB_Size)+","+SB_CSpace+","+HOG_EN+","+str(HOG_OrientCnt)+","+str(HOG_Px_Per_Cl)+","+str(HOG_Cl_Per_Blk)+",\""+HOG_Blk_Nrmlz+"\","+HOG_Trnsf_Sqrt+","+HOG_CSpace+","+str(HOG_Ch_Indx)+"], #",counter)
        counter = counter + 1