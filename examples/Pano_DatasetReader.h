#pragma once
#ifndef LDSO_DATASET_READER_H_
#define LDSO_DATASET_READER_H_

#include <sstream>
#include <fstream>
#include <dirent.h>
#include <algorithm>
#include <boost/format.hpp>

#include "Settings.h"
#include "frontend/Undistort.h"
#include "frontend/ImageRW.h"
#include "frontend/ImageAndExposure.h"

#include "internal/GlobalFuncs.h"
#include "internal/GlobalCalib.h"


#include <iostream>

using namespace std;

using namespace ldso;
using namespace ldso::internal;

inline int getdir(std::string dir, std::vector<std::string> &files) {
    DIR *dp;
    struct dirent *dirp;
    if ((dp = opendir(dir.c_str())) == NULL) {
        return -1;
    }

    while ((dirp = readdir(dp)) != NULL) {
        std::string name = std::string(dirp->d_name);
        if (name != "." && name != ".." && name.substr(name.size() - 3, name.size()) == "jpg")
            files.push_back(name);
    }
    closedir(dp);
    std::sort(files.begin(), files.end());
    if (dir.at(dir.length() - 1) != '/') dir = dir + "/";
    for (unsigned int i = 0; i < files.size(); i++) {
        if (files[i].at(0) != '/')
            files[i] = dir + files[i];
    }

    LOG(INFO) << "files size: " << files.size() << endl;
    return files.size();
}

bool cmp(std::string const &arg_a, std::string const &arg_b)
{
    return arg_a.size() < arg_b.size() || (arg_a.size() == arg_b.size() && arg_a < arg_b);
}

vector<string> getFiles(std::string cate_dir)
{
    vector<string> files;//存放文件名

    DIR *dir;
    struct dirent *ptr;
    char base[1000];

    if ((dir=opendir(cate_dir.c_str())) == NULL)
        {
        perror("Open dir error...");
                exit(1);
        }

    while ((ptr=readdir(dir)) != NULL)
    {
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir
                continue;
        else if(ptr->d_type == 8)    ///file
            //printf("d_name:%s/%s\n",basePath,ptr->d_name);
            files.push_back(cate_dir + ptr->d_name);
        else if(ptr->d_type == 10)    ///link file
            //printf("d_name:%s/%s\n",basePath,ptr->d_name);
            continue;
        else if(ptr->d_type == 4)    ///dir
        {
            files.push_back(cate_dir + ptr->d_name);
            /*
                memset(base,'\0',sizeof(base));
                strcpy(base,basePath);
                strcat(base,"/");
                strcat(base,ptr->d_nSame);
                readFileList(base);
            */
        }
    }
    closedir(dir);


    //排序，按从小到大排序
    sort(files.begin(), files.end(),cmp);
    LOG(INFO) << "files size: " << files.size() << endl;
    return files;
}

struct PrepImageItem {
    int id;
    bool isQueud;
    ImageAndExposure *pt;

    inline PrepImageItem(int _id) {
        id = _id;
        isQueud = false;
        pt = 0;
    }

    inline void release() {
        if (pt != 0) delete pt;
        pt = 0;
    }
};


class ImageFolderReader {

public:
    enum DatasetType {
        Fisheye,     // 多鱼眼模式
        Pano         // 全景模式
    };

    ImageFolderReader(DatasetType datasetType, int camNum, 
                      std::string path, std::string calibFile, std::string gammaFile,
                      std::string vignetteFile) {
        this->datasetType = datasetType;
        this->path = path;
        this->calibfile = calibFile;
        this->camNums = camNum;

        // 读取图像文件名
        if(datasetType == Fisheye)
        {
            for(size_t i= 0; i<camNum; i++)
            {
                std::string imagei = path + "/Img" + std::to_string(i) + "/";
                fisheyefiles.emplace_back(getFiles(imagei));
            }
        }
        else if (datasetType == Pano)
        {
            getdir(path, panofiles);
        }
         

        undistortMF = UndistortMultiFisheye::getUndistorterForFile(calibFile, camNum, gammaFile, vignetteFile);

        // load timestamps if possible.
        loadTimestamps();
    }

    ~ImageFolderReader() {
        delete undistortMF;
    };

    // Eigen::VectorXf getOriginalCalib() {
    //     return undistort->getOriginalParameter().cast<float>();
    // }

    // Eigen::Vector2i getOriginalDimensions() {
    //     return undistort->getOriginalSize();
    // }

    void getwh(int &w, int &h) 
    {
        w = undistortMF->getSize()[0];
        h = undistortMF->getSize()[1];
    }

    void setGlobalCalibration() {
        int w_out, h_out;
        getwh(w_out, h_out);
        //setGlobalwh(w_out, h_out);
    }

    UndistortMultiFisheye* getCalibration()
    {
        return undistortMF;
    }

    int getNumImages() {
        if(datasetType == Fisheye)
            return fisheyefiles[0].size();
        else if(datasetType == Pano)
            return panofiles.size();
    }

    double getTimestamp(int id) {
        if (timestamps.size() == 0) return id * 0.1f;
        if (id >= (int) timestamps.size()) return 0;
        if (id < 0) return 0;
        return timestamps[id];
    }


    void prepImage(int id, bool as8U = false) {

    }

    std::vector<MinimalImageB *> getImageRaw(int id) {
        return getImageRaw_internal(id, 0);
    }

    std::vector<ImageAndExposure *> getImage(int id, bool forceLoadDirectly = false) {
        return getImage_internal(id, 0);
    }


    inline float *getPhotometricGamma() {
        if (undistortMF == 0 || undistortMF->photometricUndist == 0) return 0;
        return undistortMF->photometricUndist->getG();
    }

    bool setwh(int id, std::string gammaFile, std::string vignetteFile)
    {
        std::vector<MinimalImageB *> minimg = getImageRaw_internal(id, 0);
        
        if(!bsetwh)
        {
            bsetwh =undistortMF->setwh(minimg[0], gammaFile, vignetteFile);
            undistortMF->setParasPyramids();
        }
        return true;
    }

    // undistorter. [0] always exists, [1-2] only when MT is enabled.
    UndistortMultiFisheye *undistortMF;

    bool bsetwh = false;
private:
    std::vector<MinimalImageB *> getImageRaw_internal(int id, int unused) {
    
        std::vector<MinimalImageB *> imagraw_temp;
        for(size_t i = 0; i<camNums; i++)
        {
            imagraw_temp.emplace_back(IOWrap::readImageBW_8U(fisheyefiles[i][id]));
        }
        
        return imagraw_temp;
        
    }


    std::vector<ImageAndExposure *> getImage_internal(int id, int unused) {
        std::vector<MinimalImageB *> minimg = getImageRaw_internal(id, 0);
        std::vector<ImageAndExposure *> vecret2;
        std::vector<MinimalImageB *> imagraw_temp;
        for(size_t i = 0; i<camNums; i++)
        {
            ImageAndExposure *ret2 = undistortMF->undistortFisheye<unsigned
            char>(
                    minimg[i],
                    (exposures.size() == 0 ? 1.0f : exposures[id]),
                    (timestamps.size() == 0 ? 0.0 : timestamps[id]));
            
            vecret2.emplace_back(ret2);
        }

        widthOrg = undistortMF->getOriginalSize()[0];
        heightOrg = undistortMF->getOriginalSize()[1];
        width = undistortMF->getSize()[0];
        height = undistortMF->getSize()[1];

        return vecret2;
    }


    inline void loadTimestamps() {
        LOG(INFO) << "Loading Pano timestamps!" << endl;
        std::ifstream tr;
        std::string timesFile = path + "/timestamp_FishEye.txt";

        tr.open(timesFile.c_str());
        if (!tr) {
            LOG(INFO) << "cannot find timestamp file at " << path + "/timestamp_FishEye.txt" << endl;
            return;
        }

        while (!tr.eof() && tr.good()) {

            char buf[1000] = {0};
            tr.getline(buf, 1000);

            if (buf[0] == 0)
                break;

            double stamp = atof(buf);

            if (std::isnan(stamp))
                break;

            timestamps.push_back(stamp);
        }
        tr.close();

        // // get the files
        // boost::format fmt("%s/image_0/%06d.png");
        // for (size_t i = 0; i < timestamps.size(); i++) {
        //     files.push_back((fmt % path % i).str());
        // }

        // LOG(INFO) << "Load total " << timestamps.size() << " data entries." << endl;
    }



    std::vector<ImageAndExposure *> preloadedImages;
    std::vector<std::string> panofiles;   
    std::vector<std::vector<std::string>> fisheyefiles;
    // std::vector<std::string> files;
    std::vector<double> timestamps;
    std::vector<float> exposures;
    DatasetType datasetType;

    int width, height;
    int widthOrg, heightOrg;
    int camNums;

    std::string path;
    std::string calibfile;


};

#endif // LDSO_DATASET_READER_H_
