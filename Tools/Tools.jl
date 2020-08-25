
module zt
using CSV
using DataFrames
# function testA()
#     print("A")
#     out="Out from A"
#     return out
# end
# function testB()
#     print("Call B but get out from A")
#     return testA()
# end

function GetList(pathFolder)
    fileFound=readdir(pathFolder)
    dirFileFound=readdir(pathFolder,join=true)
    return [fileFound,dirFileFound]
end

function LoadTable(pathFolder,wordSearch,combineFile)
    fileFound,dirFileFound=GetList(pathFolder)


    loadedFile=Dict()
    nFile=length(fileFound)
    global loadedFile,nFile
    for ifile=1:nFile
        global loadedFile,nFile
        println("$ifile/$nFile:",fileFound[ifile])

        loadedFile[ifile]=CSV.read(string(pathFolder,'\\',fileFound[ifile]))
    end
    return loadedFile,fileFound
end

function FilterDF(df,listFil)
    nRow,nCol=size(df)
    matIn=convert(Matrix,df)
    idxRowRemove=falses(nRow)

    if "missing" in listFil
        # index of row that has data (not missing)
        idxRowMiss=.!completecases(df)
        idxRowRemove=idxRowRemove .| idxRowMiss
    end

    for ic=1:nCol
       if "nan" in listFil
           if ic==1
               global matNan
               matNan=isnan.(matIn)
           end
           global matNan

           idxRowRemove=idxRowRemove .| matNan[:,ic]
       end

       if "zero" in listFil
           if ic==1
               global matZero
               matZero=iszero.(matIn)
           end
           global matZero
           idxRowRemove=idxRowRemove .| matZero[:,ic]
       end
   end

   return df[.!idxRowRemove,:],idxRowRemove
end # Fill DF

end # module
