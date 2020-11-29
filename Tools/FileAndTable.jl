module FileAndTable
# Please remind, you can include() other module in folder Tools 
using CSV
using DataFrames

export GetList, LoadTable, FilterDF

function GetList(pathFolder)
    fileFound=readdir(pathFolder)
    dirFileFound=readdir(pathFolder,join=true)
    return [fileFound,dirFileFound]
end

function FindFile(pathSearch)
    fileFound=readdir(pathFolder)
    dirFileFound=readdir(pathFolder,join=true)


end

function LoadTable(pathFolder,wordSearch,combineFile)
    fileFound,dirFileFound=GetList(pathFolder)

    loadedFile=Dict()
    listLoaded=Dict()
    nFile=length(fileFound)
    countFile=0
    global loadedFile,listLoaded,nFile,countFile
    for ifile=1:nFile
        global loadedFile
        # println("$ifile/$nFile:",fileFound[ifile])

        if occursin(wordSearch,fileFound[ifile])
            countFile=countFile+1;    
            listLoaded[countFile]=fileFound[ifile];
            loadedFile[countFile]=CSV.read(string(pathFolder,'\\',fileFound[ifile]));
        end
    end
    return loadedFile,listLoaded
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
