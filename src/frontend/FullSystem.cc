#include "Feature.h"
#include "Frame.h"
#include "Point.h"

#include "frontend/FullSystem.h"
#include "frontend/CoarseInitializer.h"
#include "frontend/CoarseTracker.h"
#include "frontend/LoopClosing.h"

#include "internal/ImmaturePoint.h"
#include "internal/GlobalCalib.h"
#include "internal/GlobalFuncs.h"
#include "internal/OptimizationBackend/EnergyFunctional.h"

#include <opencv2/features2d/features2d.hpp>
#include <iomanip>
#include <opencv2/highgui/highgui.hpp>

using namespace ldso;
using namespace ldso::internal;

namespace ldso {

    FullSystem::FullSystem(shared_ptr<ORBVocabulary> voc) :
        coarseDistanceMap(new CoarseDistanceMap(wG[0], hG[0])),
        coarseTracker(new CoarseTracker(wG[0], hG[0])),
        coarseTracker_forNewKF(new CoarseTracker(wG[0], hG[0])),
        coarseInitializer(new CoarseInitializer(wG[0], hG[0])),
        ef(new EnergyFunctional()),
        Hcalib(new Camera(fxG[0], fyG[0], cxG[0], cyG[0])),
        globalMap(new Map(this)),
        vocab(voc) {

        // LOG(INFO) << "This is Pano-DSO proposed by Group of  Photogrammetry and Computer Vision of Wuhan University, 
        //      This is Direct Sparse Odometry, a fully direct VO proposed by TUM vision group."
        //     "For more information about dso, see Direct Sparse Odometry, J. Engel, V. Koltun, "
        //     "D. Cremers, In arXiv:1607.02565, 2016. For loop closing part, see "    
        //     "LDSO: Direct Sparse Odometry with Loop Closure, X. Gao, R. Wang, N. Demmel, D. Cremers, "
        //     "In International Conference on Intelligent Robots and Systems (IROS), 2018 " << endl;

        Hcalib->CreateCH(Hcalib);   // 内参赋值，求逆外参 gamma系数
        lastCoarseRMSE.setConstant(100);
        ef->red = &this->threadReduce;
        mappingThread = thread(&FullSystem::mappingLoop, this);

        pixelSelector = shared_ptr<PixelSelector>(new PixelSelector(wG[0], hG[0]));
        selectionMap = new float[wG[0] * hG[0]];

        if (setting_enableLoopClosing) {
            loopClosing = shared_ptr<LoopClosing>(new LoopClosing(this));
            if (setting_fastLoopClosing)
                LOG(INFO) << "Use fast loop closing" << endl;
        } else {
            LOG(INFO) << "loop closing is disabled" << endl;
        }

        minIdJetVisTracker = -1;
	    maxIdJetVisTracker = -1;

    }

    // multi-fisheye FullSystem 初始化
    FullSystem::FullSystem(shared_ptr<ORBVocabulary> voc, UndistortMultiFisheye *uMF) :
        coarseDistanceMap(new CoarseDistanceMap(uMF)),
        coarseTracker(new CoarseTracker(uMF)),
        coarseTracker_forNewKF(new CoarseTracker(uMF)),
        coarseInitializer(new CoarseInitializer(uMF)),
        ef(new EnergyFunctional(uMF)),
        //Hcalib(new Camera()),
        globalMap(new Map(this)),
        vocab(voc), 
        UMF(uMF)
    {

        cout << 
            "===============================================================================================================================" << endl << endl <<
            "This is Pano-DSO, a Multi fisheye direct VO proposed by Group of Photogrammetry and Computer Vision of Wuhan University." << endl <<
            "For more information about pano-dso, see  " << endl <<
            "This is Direct Sparse Odometry, a fully direct VO proposed by TUM vision group." << endl <<
            "For more information about dso, see Direct Sparse Odometry, J. Engel, V. Koltun, D. Cremers, In arXiv:1607.02565, 2016. " << endl << endl <<
            "===============================================================================================================================" << endl;
            // "For loop closing part, see  LDSO: Direct Sparse Odometry with Loop Closure, X. Gao, R. Wang, N. Demmel, D. Cremers, "
            // "In International Conference on Intelligent Robots and Systems (IROS), 2018 " << endl;

        //Hcalib->CreateCH(Hcalib);   // 内参赋值，求逆外参 gamma系数
        multicamera_nums = UMF->camNums;
        lastCoarseRMSE.setConstant(100);
        ef->red = &this->threadReduce;
        int w = UMF->getSize()(0);
        int h = UMF->getSize()(1);
        mappingThread = thread(&FullSystem::mappingLoopforMF, this);

        pixelSelector = shared_ptr<PixelSelector>(new PixelSelector(w, h));
        selectionMap = new float[w * h];

        if (setting_enableLoopClosing) {
            loopClosing = shared_ptr<LoopClosing>(new LoopClosing(this));
            if (setting_fastLoopClosing)
                LOG(INFO) << "Use fast loop closing" << endl;
        } else {
            LOG(INFO) << "loop closing is disabled" << endl;
        }

        minIdJetVisTracker = -1;
	    maxIdJetVisTracker = -1;

    }

    FullSystem::~FullSystem() {
        blockUntilMappingIsFinished();
        // remember to release the inner structure
        this->unmappedTrackedFrames.clear();
        if (setting_enableLoopClosing == false) {
            delete[] selectionMap;
        } else {
        }
    }

    void FullSystem::addActiveFrame(ImageAndExposure *image, int id) {
        if (isLost)
            return;
        unique_lock<mutex> lock(trackMutex);

        LOG(INFO) << "*** taking frame " << id << " ***" << endl;

        // create frame and frame hessian
        shared_ptr<Frame> frame(new Frame(image->timestamp));
        frame->CreateFH(frame);
        allFrameHistory.push_back(frame);

        // ==== make images ==== //
        shared_ptr<FrameHessian> fh = frame->frameHessian;
        fh->ab_exposure = image->exposure_time;
        fh->makeImages(image->image, Hcalib->mpCH);

        if (!initialized) {
            LOG(INFO) << "Initializing ... " << endl;
            // use initializer
            if (coarseInitializer->frameID < 0) {   // first frame not set, set it
                coarseInitializer->setFirst(Hcalib->mpCH, fh);   // 初始化-设置第一帧：生成金字塔各层内参，提取点(局部梯度最大)，初始化点的逆深度为1,通过构建kdtree来找到每个点在像素平面上的相邻点,frameID=0
            } else if (coarseInitializer->trackFrame(fh)) {   // 
                // init succeeded
                initializeFromInitializer(fh);
                lock.unlock();
                deliverTrackedFrame(fh, true);  // create new keyframe
                LOG(INFO) << "init success." << endl;
            } else {
                // still initializing
                frame->poseValid = false;
                frame->ReleaseAll();        // don't need this frame, release all the internal
            }
            return;
        } else {
            // init finished, do tracking
            // =========================== SWAP tracking reference?. =========================
            if (coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID) {
                unique_lock<mutex> crlock(coarseTrackerSwapMutex);
                LOG(INFO) << "swap coarse tracker to " << coarseTracker_forNewKF->refFrameID << endl;
                auto tmp = coarseTracker;
                coarseTracker = coarseTracker_forNewKF;
                coarseTracker_forNewKF = tmp;
            }

            // track the new frame and get the state
            LOG(INFO) << "tracking new frame" << endl;
            Vec4 tres = trackNewCoarse(fh);

            if (!std::isfinite((double) tres[0]) || !std::isfinite((double) tres[1]) ||
                !std::isfinite((double) tres[2]) || !std::isfinite((double) tres[3])) {
                // invalid result
                LOG(WARNING) << "Initial Tracking failed: LOST!" << endl;
                isLost = true;
                return;
            }

            bool needToMakeKF = false;
            if (setting_keyframesPerSecond > 0) {
                // make key frame by time
                needToMakeKF = allFrameHistory.size() == 1 ||
                               (frame->timeStamp - frames.back()->timeStamp) >
                               0.95f / setting_keyframesPerSecond;
            } else {
                Vec2 refToFh = AffLight::fromToVecExposure(coarseTracker->lastRef->ab_exposure, fh->ab_exposure,
                                                           coarseTracker->lastRef_aff_g2l, fh->aff_g2l());
                // 关键帧的选择主要考虑当前帧和前一关键帧在点的光流变化，不考虑旋转情况下的光流变化，曝光参数的变化，三者加权相加大于１时新建关键帧。
                float b = setting_kfGlobalWeight * setting_maxShiftWeightT * sqrtf((double) tres[1]) /
                          (wG[0] + hG[0]) +
                          setting_kfGlobalWeight * setting_maxShiftWeightR * sqrtf((double) tres[2]) /
                          (wG[0] + hG[0]) +
                          setting_kfGlobalWeight * setting_maxShiftWeightRT * sqrtf((double) tres[3]) /
                          (wG[0] + hG[0]) +
                          setting_kfGlobalWeight * setting_maxAffineWeight * fabs(logf((float) refToFh[0]));

                bool b1 = b > 1;
                bool b2 = 2 * coarseTracker->firstCoarseRMSE < tres[0];

                needToMakeKF = allFrameHistory.size() == 1 || b1 || b2;
            }

            if (viewer)
                viewer->publishCamPose(fh->frame, Hcalib->mpCH);

            lock.unlock();
            LOG(INFO) << "deliver frame " << fh->frame->id << endl;
            deliverTrackedFrame(fh, needToMakeKF);
            LOG(INFO) << "add active frame returned" << endl << endl;
            return;
        }
    }

    // for multi-fisheye image
    void FullSystem::addActiveFrame(vector<ImageAndExposure *> image, int id) {
        if (isLost)
            return;
        unique_lock<mutex> lock(trackMutex);

        LOG(INFO) << "*** taking frame " << id << " ***" << endl;

        // create frame and frame hessian
        shared_ptr<Frame> frame(new Frame(image[0]->timestamp));
        frame->CreateFH(frame, UMF);
        allFrameHistory.push_back(frame);

        // ==== make images ==== //
        shared_ptr<FrameHessian> fh = frame->frameHessian;
        fh->ab_exposure = image[0]->exposure_time;
        vector<float *>  vimage;
        for(int i=0; i<UMF->getcamNums(); i++)
            vimage.emplace_back(image[i]->image);
        fh->makeImages(vimage, UMF);

        if(!initialized) 
        {
            LOG(INFO) << "Initializing ... " << endl;

            if(coarseInitializer->frameID < 0)
            {
                LOG(INFO)  << "set first frame" << endl;
                // for(int n = 0; n < UMF->camNums; n++)
                // {
                //     float *mapmax0 = fh->vabsSquaredGrad[0][n];

                //     ofstream outim;
                //     string path = "/home/jiangfan/桌面/pan_dso_calib/pano_tracking/vabsSquaredGrad_coarseInitializer_" + to_string(n) + ".txt";
                //     outim.open(path.c_str());

                //     int w = UMF->getOriginalSize()(0);
                //     int h = UMF->getOriginalSize()(1);

                //     for (int iw = 0; iw < w;  iw++)
                //         for (int ih = 0; ih < h;  ih++)
                //     {
                //         int idx = iw + ih * h;
                //         if(mapmax0[idx] < 0)
                //         {
                //             int aa = 0;
                //         }
                        
                //         outim << iw << " " << ih << " " << mapmax0[idx] << " " << fh->vabsSquaredGrad[0][n][idx] <<endl;
                //     }
                //     outim.close();
                // }
                coarseInitializer->setFirstforMF(fh);
                LOG(INFO) << "fisrt is over" << endl;
            }
            else if(coarseInitializer->trackFrameforMF(fh, viewer))
            {
                initializeFromInitializerforMF(fh);     
                lock.unlock();
                deliverTrackedFrameforMF(fh, true);  
                LOG(INFO) << "Init succes!" << endl;
            }
            else
            {
                frame->poseValid = false;
                frame->ReleaseAll();
            }

            return;
        }
        else
        {
            // init finished, do tracking
            // =========================== SWAP tracking reference?. =========================
            if (coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID) 
            {
                unique_lock<mutex> crlock(coarseTrackerSwapMutex);
                LOG(INFO) << "swap coarse tracker to " << coarseTracker_forNewKF->refFrameID << endl;
                auto tmp = coarseTracker;
                coarseTracker = coarseTracker_forNewKF;
                coarseTracker_forNewKF = tmp;
            }

            // track the new frame and get the state
            LOG(INFO) << "tracking new frame" << endl;
            Vec4 tres = trackNewCoarseforMF(fh);

            if (!std::isfinite((double) tres[0]) || !std::isfinite((double) tres[1]) ||
                !std::isfinite((double) tres[2]) || !std::isfinite((double) tres[3])) 
            {
                // invalid result
                LOG(WARNING) << "Initial Tracking failed: LOST!" << endl;
                isLost = true;
                return;
            }

            bool needToMakeKF = false;
            if (setting_keyframesPerSecond > 0) 
            {
                // make key frame by time
                needToMakeKF = allFrameHistory.size() == 1 ||
                               (frame->timeStamp - frames.back()->timeStamp) >
                               0.95f / setting_keyframesPerSecond;
            } 
            else 
            {
                float refToFh_a = 0;
                for(int n1 = 0; n1 < UMF->camNums; n1++)
                {
                    for(int n2 = 0; n2 < UMF->camNums; n2++)
                    {
                        Vec2 refToFh = AffLight::fromToVecExposure(coarseTracker->lastRef->ab_exposure, fh->ab_exposure,
                                                           coarseTracker->vlastRef_aff_g2l[n1], fh->aff_g2lforMF()[n2]);
                        refToFh_a += fabs(logf(refToFh[0]));
                    }
                }
                refToFh_a = refToFh_a / (UMF->camNums * UMF->camNums);
                // 关键帧的选择主要考虑当前帧和前一关键帧在点的光流变化，不考虑旋转情况下的光流变化，曝光参数的变化，三者加权相加大于１时新建关键帧。    
                float b = setting_kfGlobalWeight * setting_maxShiftWeightTSphere * sqrtf((double) tres[1])       // 平移像素位移 
                            +
                          setting_kfGlobalWeight * setting_maxShiftWeightRSphere * sqrtf((double) tres[2])       // 0
                            +
                          setting_kfGlobalWeight * setting_maxShiftWeightRTSphere * sqrtf((double) tres[3])      // 旋转+平移像素位移
                            +
                          setting_kfGlobalWeight * setting_maxAffineWeight * refToFh_a;                     // 光度变化大

                bool b1 = b > 1;
                bool b2 = 2 * coarseTracker->firstCoarseRMSE < tres[0];                                     // 误差能量变化太大(最初的两倍)

                needToMakeKF = allFrameHistory.size() == 1 || b1 || b2;
            }

            if (viewer)
                viewer->publishCamPoseforMF(fh->frame, UMF);

            lock.unlock();
            LOG(INFO) << "deliver frame " << fh->frame->id << endl;
            // deliverTrackedFrameforMF(fh, needToMakeKF);
            deliverTrackedFrameforMF(fh, true);
            LOG(INFO) << "add active frame returned" << endl << endl;
            return;
        }
    }

    void FullSystem::deliverTrackedFrame(shared_ptr<FrameHessian> fh, bool needKF) {
        if (linearizeOperation) {
            if (needKF) {
                makeKeyFrame(fh);
            } else {
                makeNonKeyFrame(fh);
            }
        } else {
            unique_lock<mutex> lock(trackMapSyncMutex);
            unmappedTrackedFrames.push_back(fh->frame);
            trackedFrameSignal.notify_all();
            while (coarseTracker_forNewKF->refFrameID == -1 && coarseTracker->refFrameID == -1) {
                LOG(INFO) << "wait for mapped frame signal" << endl;
                mappedFrameSignal.wait(lock);
            }
            lock.unlock();
        }
    }

    // for multi-fisheye 
    void FullSystem::deliverTrackedFrameforMF(shared_ptr<FrameHessian> fh, bool needKF)
    {
        if (linearizeOperation) 
        {
            if (needKF) {
                makeKeyFrameforMF(fh);
            } else {
                makeNonKeyFrameforMF(fh);
            }
        } else 
        {
            unique_lock<mutex> lock(trackMapSyncMutex);
            unmappedTrackedFrames.push_back(fh->frame);
            trackedFrameSignal.notify_all();
            while (coarseTracker_forNewKF->refFrameID == -1 && coarseTracker->refFrameID == -1) {
                LOG(INFO) << "wait for mapped frame signal" << endl;
                mappedFrameSignal.wait(lock);
            }
            lock.unlock();
        }
    }

    Vec4 FullSystem::trackNewCoarse(shared_ptr<FrameHessian> fh) {

        assert(allFrameHistory.size() > 0);

        shared_ptr<FrameHessian> lastF = coarseTracker->lastRef;
        CHECK(coarseTracker->lastRef->frame != nullptr);

        AffLight aff_last_2_l = AffLight(0, 0);

        // try a lot of pose values and see which is the best
        std::vector<SE3, Eigen::aligned_allocator<SE3>>
            lastF_2_fh_tries;
        if (allFrameHistory.size() == 2)
            for (unsigned int i = 0; i < lastF_2_fh_tries.size(); i++)  // TODO: maybe wrong, size is obviously zero
                lastF_2_fh_tries.push_back(SE3());  // use identity
        else {

            // fill the pose tries ...
            // use the last before last and the last before before last (well my English is really poor...)
            shared_ptr<Frame> slast = allFrameHistory[allFrameHistory.size() - 2];
            shared_ptr<Frame> sprelast = allFrameHistory[allFrameHistory.size() - 3];

            SE3 slast_2_sprelast;
            SE3 lastF_2_slast;

            {    // lock on global pose consistency!
                unique_lock<mutex> crlock(shellPoseMutex);
                slast_2_sprelast = sprelast->getPose() * slast->getPose().inverse();
                lastF_2_slast = slast->getPose() * lastF->frame->getPose().inverse();
                aff_last_2_l = slast->aff_g2l;
            }
            SE3 fh_2_slast = slast_2_sprelast;// assumed to be the same as fh_2_slast.

            // get last delta-movement.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast);    // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() *
                                       lastF_2_slast);    // assume double motion (frame skipped)
            lastF_2_fh_tries.push_back(
                SE3::exp(fh_2_slast.log() * 0.5).inverse() * lastF_2_slast); // assume half motion.
            lastF_2_fh_tries.push_back(lastF_2_slast); // assume zero motion.
            lastF_2_fh_tries.push_back(SE3()); // assume zero motion FROM KF.


            // just try a TON of different initializations (all rotations). In the end,
            // if they don't work they will only be tried on the coarsest level, which is super fast anyway.
            // also, if tracking rails here we loose, so we really, really want to avoid that.
            for (float rotDelta = 0.02;
                 rotDelta < 0.05; rotDelta += 0.01) {    // TODO changed this into +=0.01 where DSO writes ++
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, rotDelta, 0, 0),
                                               Vec3(0, 0, 0)));            // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, 0, rotDelta, 0),
                                               Vec3(0, 0, 0)));            // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, 0, 0, rotDelta),
                                               Vec3(0, 0, 0)));            // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, -rotDelta, 0, 0),
                                               Vec3(0, 0, 0)));            // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, 0, -rotDelta, 0),
                                               Vec3(0, 0, 0)));            // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, 0, 0, -rotDelta),
                                               Vec3(0, 0, 0)));            // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, 0),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, 0, rotDelta, rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, rotDelta, 0, rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, 0),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, 0, -rotDelta, rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, -rotDelta, 0, rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, 0),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, 0, rotDelta, -rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, rotDelta, 0, -rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, 0),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, 0, -rotDelta, -rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, -rotDelta, 0, -rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, -rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, -rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, -rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, -rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
            }

            if (!slast->poseValid || !sprelast->poseValid || !lastF->frame->poseValid) {
                lastF_2_fh_tries.clear();
                lastF_2_fh_tries.push_back(SE3());
            }
        }

        Vec3 flowVecs = Vec3(100, 100, 100);
        SE3 lastF_2_fh = SE3();
        AffLight aff_g2l = AffLight(0, 0);

        // as long as maxResForImmediateAccept is not reached, I'll continue through the options.
        // I'll keep track of the so-far best achieved residual for each level in achievedRes.
        // If on a coarse level, tracking is WORSE than achievedRes, we will not continue to save time.

        Vec5 achievedRes = Vec5::Constant(NAN);
        bool haveOneGood = false;
        int tryIterations = 0;
        for (unsigned int i = 0; i < lastF_2_fh_tries.size(); i++) {

            AffLight aff_g2l_this = aff_last_2_l;
            SE3 lastF_2_fh_this = lastF_2_fh_tries[i];

            // use coarse tracker to solve the iteration
            bool trackingIsGood = coarseTracker->trackNewestCoarse(
                fh, lastF_2_fh_this, aff_g2l_this,
                pyrLevelsUsed - 1,
                achievedRes);    // in each level has to be at least as good as the last try.
            tryIterations++;

            // do we have a new winner?
            if (trackingIsGood && std::isfinite((float) coarseTracker->lastResiduals[0]) &&
                !(coarseTracker->lastResiduals[0] >= achievedRes[0])) {
                flowVecs = coarseTracker->lastFlowIndicators;
                aff_g2l = aff_g2l_this;
                lastF_2_fh = lastF_2_fh_this;
                haveOneGood = true;
            }

            // take over achieved res (always).
            if (haveOneGood) {
                for (int i = 0; i < 5; i++) {
                    if (!std::isfinite((float) achievedRes[i]) ||
                        achievedRes[i] >
                        coarseTracker->lastResiduals[i])    // take over if achievedRes is either bigger or NAN.
                        achievedRes[i] = coarseTracker->lastResiduals[i];
                }
            }

            if (haveOneGood && achievedRes[0] < lastCoarseRMSE[0] * setting_reTrackThreshold)   // lastCoarseRMSE 初始值为100
                break;
        }

        if (!haveOneGood) {
            LOG(WARNING) << "BIG ERROR! tracking failed entirely. Take predicted pose and hope we may somehow recover."
                         << endl;
            flowVecs = Vec3(0, 0, 0);
            aff_g2l = aff_last_2_l;
            lastF_2_fh = lastF_2_fh_tries[0];
        }

        lastCoarseRMSE = achievedRes;  //平均残差   开方

        // set the pose of new frame
        CHECK(coarseTracker->lastRef->frame != nullptr);
        SE3 camToWorld = lastF->frame->getPose().inverse() * lastF_2_fh.inverse();
        fh->frame->setPose(camToWorld.inverse());
        fh->frame->aff_g2l = aff_g2l;

        if (coarseTracker->firstCoarseRMSE < 0)
            coarseTracker->firstCoarseRMSE = achievedRes[0];

        LOG(INFO) << "Coarse Tracker tracked ab = " << aff_g2l.a << " " << aff_g2l.b << " (exp " << fh->ab_exposure
                  << " ). Res " << achievedRes[0] << endl;

        return Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
    }

    // for multi-fisheye
    Vec4 FullSystem::trackNewCoarseforMF(shared_ptr<FrameHessian> fh)
    {
        assert(allFrameHistory.size() > 0);

        shared_ptr<FrameHessian> lastF = coarseTracker->lastRef;    // 参考帧
        CHECK(coarseTracker->lastRef->frame != nullptr);

        vector<AffLight> vaff_last_2_l(UMF->camNums, AffLight(0, 0));

        // try a lot of pose values and see which is the best
        std::vector<SE3, Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;   // 前一关键帧到当前帧的相对位姿的初值
        if (allFrameHistory.size() == 2)
        {    
            // for (unsigned int i = 0; i < lastF_2_fh_tries.size(); i++)  // TODO: maybe wrong, size is obviously zero
            for (unsigned int i = 0; i < 2; i++)  // TODO: maybe wrong, size is obviously zero
                lastF_2_fh_tries.push_back(SE3());  // use identity
        }  
        else
        {
            // fill the pose tries ...
            // use the last before last and the last before before last (well my English is really poor...)
            shared_ptr<Frame> slast = allFrameHistory[allFrameHistory.size() - 2];  // 最后一帧的前一帧 倒数第二帧
            shared_ptr<Frame> sprelast = allFrameHistory[allFrameHistory.size() - 3];   // 倒数第三帧

            SE3 slast_2_sprelast;
            SE3 lastF_2_slast;

            {    // lock on global pose consistency!
                unique_lock<mutex> crlock(shellPoseMutex);
                slast_2_sprelast = sprelast->getPose() * slast->getPose().inverse();        // T n-2_n-1
                lastF_2_slast = slast->getPose() * lastF->frame->getPose().inverse();       // T n-1_ref
                vaff_last_2_l = slast->vaff_g2l;
            }
            SE3 fh_2_slast = slast_2_sprelast;// assumed to be the same as fh_2_slast. // 当前帧到上一帧 = 上一帧和上上一帧的    Tn-1_n = Tn-2_n-1
        
            // get last delta-movement.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast);    // assume constant motion. 前一关键帧到当前帧
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast);    // assume double motion (frame skipped)
            lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log() * 0.5).inverse() * lastF_2_slast); // assume half motion.
            lastF_2_fh_tries.push_back(lastF_2_slast); // assume zero motion.
            lastF_2_fh_tries.push_back(SE3()); // assume zero motion FROM KF.

            // 尝试不同的旋转变动
            // just try a TON of different initializations (all rotations). In the end,
            // if they don't work they will only be tried on the coarsest level, which is super fast anyway.
            // also, if tracking rails here we loose, so we really, really want to avoid that.
            for (float rotDelta = 0.02; rotDelta < 0.05; rotDelta += 0.01) 
            {    // TODO changed this into +=0.01 where DSO writes ++
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, rotDelta, 0, 0),
                                               Vec3(0, 0, 0)));            // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, 0, rotDelta, 0),
                                               Vec3(0, 0, 0)));            // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, 0, 0, rotDelta),
                                               Vec3(0, 0, 0)));            // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, -rotDelta, 0, 0),
                                               Vec3(0, 0, 0)));            // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, 0, -rotDelta, 0),
                                               Vec3(0, 0, 0)));            // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, 0, 0, -rotDelta),
                                               Vec3(0, 0, 0)));            // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, 0),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, 0, rotDelta, rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, rotDelta, 0, rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, 0),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, 0, -rotDelta, rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, -rotDelta, 0, rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, 0),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, 0, rotDelta, -rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, rotDelta, 0, -rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, 0),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, 0, -rotDelta, -rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, -rotDelta, 0, -rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, -rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, -rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, -rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, -rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, rotDelta),
                                               Vec3(0, 0, 0)));    // assume constant motion.
            }

            if (!slast->poseValid || !sprelast->poseValid || !lastF->frame->poseValid) 
            {
                lastF_2_fh_tries.clear();
                lastF_2_fh_tries.push_back(SE3());
            }
        }

        Vec3 flowVecs = Vec3(12, 12, 12);  // 100 像素对应的大概弧长
        SE3 lastF_2_fh = SE3();
        // AffLight aff_g2l = AffLight(0, 0);
        vector<AffLight> vaff_g2l(UMF->camNums, AffLight(0, 0));

        // 把到目前为止最好的残差值作为每一层的阈值 粗层的能量值大, 也不继续优化了, 来节省时间

        // as long as maxResForImmediateAccept is not reached, I'll continue through the options.
        // I'll keep track of the so-far best achieved residual for each level in achievedRes.
        // If on a coarse level, tracking is WORSE than achievedRes, we will not continue to save time.

        Vec5 achievedRes = Vec5::Constant(NAN);
        bool haveOneGood = false;
        int tryIterations = 0;
        // 逐个试验
        for (unsigned int i = 0; i < lastF_2_fh_tries.size(); i++)
        {
            cout << "第/总" << i << "/" << lastF_2_fh_tries.size() << endl;
            vector<AffLight> vaff_g2l_this = vaff_last_2_l;  //  上一帧的赋值当前帧
            SE3 lastF_2_fh_this = lastF_2_fh_tries[i];

            // use coarse tracker to solve the iteration
            bool trackingIsGood = coarseTracker->trackNewestCoarseforMF(fh, lastF_2_fh_this, vaff_g2l_this,
                                                                    pyrLevelsUsed - 1,achievedRes);    // in each level has to be at least as good as the last try.
            tryIterations++;
            // 如果跟踪正常, 并且0层残差比最好的还好留下位姿, 保存最好的每一层的能量值
            // do we have a new winner?
            if (trackingIsGood && std::isfinite((float) coarseTracker->lastResiduals[0]) &&
                !(coarseTracker->lastResiduals[0] >= achievedRes[0])) 
            {
                flowVecs = coarseTracker->lastFlowIndicators;
                vaff_g2l = vaff_g2l_this;
                lastF_2_fh = lastF_2_fh_this;
                haveOneGood = true;
            }

            // take over achieved res (always). 保存Res中新旧最小的　到　achieveRes
            if (haveOneGood) 
            {
                for (int i = 0; i < 5; i++) 
                {
                    if (!std::isfinite((float) achievedRes[i]) ||
                        achievedRes[i] > coarseTracker->lastResiduals[i])    // take over if achievedRes is either bigger or NAN.
                        achievedRes[i] = coarseTracker->lastResiduals[i];
                }
            }
            // 1.如果此次残差小于前一次试验位姿的最小残差(haveOneGood)  2.并且此次残差小于阈值(前一次跟踪中最小残差*1.5)则暂停, 并且为下次设置阈值
            if (haveOneGood && achievedRes[0] < lastCoarseRMSE[0] * setting_reTrackThreshold)   // lastCoarseRMSE 初始值为100
                break;
        }

        if (!haveOneGood)   //  跟踪失败
        {
            LOG(WARNING) << "BIG ERROR! tracking failed entirely. Take predicted pose and hope we may somehow recover."
                         << endl;
            flowVecs = Vec3(0, 0, 0);
            vaff_g2l = vaff_last_2_l;
            lastF_2_fh = lastF_2_fh_tries[0];
        }

        // 把这次得到的最好值给下次用来当阈值
        lastCoarseRMSE = achievedRes;  //平均残差   开方

        // set the pose of new frame
        CHECK(coarseTracker->lastRef->frame != nullptr);
        SE3 camToWorld = lastF->frame->getPose().inverse() * lastF_2_fh.inverse();
        fh->frame->setPose(camToWorld.inverse());
        fh->frame->vaff_g2l = vaff_g2l;

        if (coarseTracker->firstCoarseRMSE < 0)
            coarseTracker->firstCoarseRMSE = achievedRes[0];

        LOG(INFO) << "Coarse Tracker tracked ab = " << vaff_g2l[0].a << " " << vaff_g2l[0].b << " (exp " << fh->ab_exposure
                  << " ). Res " << achievedRes[0] << endl;

        return Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
    }

    void FullSystem::blockUntilMappingIsFinished() {
        {
            unique_lock<mutex> lock(trackMapSyncMutex);
            if (!runMapping) {
                // mapping is already finished, no need to finish again
                return;
            }
            runMapping = false;
            trackedFrameSignal.notify_all();
        }

        mappingThread.join();

        if (setting_enableLoopClosing)
            loopClosing->SetFinish(true);

        // Update world points in case optimization hasn't run (with all keyframes)
        // It would maybe be better if the 3d points would always be updated as soon
        // as the poses or depths are updated (no matter if in PGO or in sliding window BA)
        globalMap->UpdateAllWorldPoints();
    }

    void FullSystem::makeKeyFrame(shared_ptr<FrameHessian> fh) {

        shared_ptr<Frame> frame = fh->frame;
        auto refFrame = frames.back();

        {
            unique_lock<mutex> crlock(shellPoseMutex);
            fh->setEvalPT_scaled(fh->frame->getPose(), fh->frame->aff_g2l);
            fh->frame->setPoseOpti(Sim3(fh->frame->getPose().matrix()));
        }

        LOG(INFO) << "frame " << fh->frame->id << " is marked as key frame, active keyframes: " << frames.size()
                  << endl;

        // trace new keyframe 利用当前帧对前面关键帧中的未成熟点进行逆深度更新
        traceNewCoarse(fh);

        unique_lock<mutex> lock(mapMutex);

        // == flag frames to be marginalized  == //  标记后面需要边缘化（从活动窗口踢出）的帧
        flagFramesForMarginalization(fh);

        // add new frame to hessian struct
        {
            unique_lock<mutex> lck(framesMutex);
            fh->idx = frames.size();
            frames.push_back(fh->frame);       // 将当前帧加入到滑动窗口中
            fh->frame->kfId = fh->frameID = globalMap->NumFrames();
        }

        ef->insertFrame(fh, Hcalib->mpCH);   // 将当前帧加入到总的能量函数中EnergyFunctional（ef）
        setPrecalcValues();    // 计算一下该窗口中其他帧与当前帧之间的一些参数比如相对光度、距离

        // =========================== add new residuals for old points =========================
        LOG(INFO) << "adding new residuals" << endl;
        int numFwdResAdde = 0;
        for (auto fht : frames) { // go through all active frames  遍历窗口中之前所有帧
            shared_ptr<FrameHessian> &fh1 = fht->frameHessian;
            if (fh1 == fh)
                continue;
            for (auto feat: fht->features) {    // 遍历窗口中之前所有帧的成熟点
                if (feat->status == Feature::FeatureStatus::VALID
                    && feat->point->status == Point::PointStatus::ACTIVE) {

                    shared_ptr<PointHessian> ph = feat->point->mpPH;

                    // add new residuals into this point hessian
                    shared_ptr<PointFrameResidual> r(
                        new PointFrameResidual(ph, fh1, fh));    // residual from fh1 to fh

                    r->setState(ResState::IN);
                    ph->residuals.push_back(r);
                    ef->insertResidual(r);      //  构建它们和新的关键帧的点帧误差PointFrameResidual，加入到ef中

                    ph->lastResiduals[1] = ph->lastResiduals[0];
                    ph->lastResiduals[0] = pair<shared_ptr<PointFrameResidual>, ResState>(r, ResState::IN);
                    numFwdResAdde++;
                }
            }
        }

        // =========================== Activate Points (& flag for marginalization). =========================
        activatePointsMT();    // 激活窗口中之前所有帧中符合条件的未成熟点，将其加入到ef中FullSystem::activatePointsMT
        ef->makeIDX();

        // =========================== OPTIMIZE ALL =========================
        fh->frameEnergyTH = frames.back()->frameHessian->frameEnergyTH;
        LOG(INFO) << "call optimize on kf " << frame->kfId << endl;
        float rmse = optimize(setting_maxOptIterations);    // 对活动窗口中的所有变量进行优化
        LOG(INFO) << "optimize is done!" << endl;

        // =========================== Figure Out if INITIALIZATION FAILED =========================
        int numKFs = globalMap->NumFrames();
        // if (numKFs <= 4) {
        //     if (numKFs == 2 && rmse > 20 * benchmark_initializerSlackFactor) {
        //         LOG(WARNING) << "I THINK INITIALIZATINO FAILED! Resetting." << endl;
        //         LOG(INFO) << "rmse = " << rmse << endl;
        //         initFailed = true;
        //     }
        //     if (numKFs == 3 && rmse > 13 * benchmark_initializerSlackFactor) {
        //         LOG(WARNING) << "I THINK INITIALIZATINO FAILED! Resetting." << endl;
        //         LOG(INFO) << "rmse = " << rmse << endl;
        //         initFailed = true;
        //     }
        //     if (numKFs == 4 && rmse > 9 * benchmark_initializerSlackFactor) {
        //         LOG(WARNING) << "I THINK INITIALIZATINO FAILED! Resetting." << endl;
        //         LOG(INFO) << "rmse = " << rmse << endl;
        //         initFailed = true;
        //     }
        // }

        if(numKFs <= 4)
	    {
		if(numKFs==2 && rmse > 25*benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
		if(numKFs==3 && rmse > 18*benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
		if(numKFs==4 && rmse > 13*benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
	    }

        if (isLost)
            return;

        // =========================== REMOVE OUTLIER =========================
        removeOutliers();     // 去除外点

        // swap the coarse Tracker for new kf
        {
            unique_lock<mutex> crlock(coarseTrackerSwapMutex);
            coarseTracker_forNewKF->makeK(Hcalib->mpCH);
            vector<shared_ptr<FrameHessian >> fhs;
            for (auto &f: frames) fhs.push_back(f->frameHessian);
            coarseTracker_forNewKF->setCoarseTrackingRef(fhs);

            coarseTracker_forNewKF->debugPlotIDepthMap(&minIdJetVisTracker, &maxIdJetVisTracker, viewer);
        }

        // =========================== (Activate-)Marginalize Points =========================
        // traditional bundle adjustment when marging all points
        flagPointsForRemoval();   
        ef->dropPointsF();

        getNullspaces(
            ef->lastNullspaces_pose,
            ef->lastNullspaces_scale,
            ef->lastNullspaces_affA,
            ef->lastNullspaces_affB);

        ef->marginalizePointsF();      // 边缘化不需要的点

        // =========================== add new Immature points & new residuals =========================
        makeNewTraces(fh, 0);    // 在当前帧中提取未成熟点

        // record the relative poses, note we are building a covisibility graph in fact
        auto minandmax = std::minmax_element(frames.begin(), frames.end(), CmpFrameKFID());
        unsigned long minKFId = (*minandmax.first)->kfId;
        unsigned long maxKFId = (*minandmax.second)->kfId;

        if (setting_fastLoopClosing == false) {
            // record all active keyframes
            for (auto &fr : frames) {
                auto allKFs = globalMap->GetAllKFs();
                for (auto &f2: allKFs) {
                    if (f2->kfId > minKFId && f2->kfId < maxKFId && f2 != fr) {
                        unique_lock<mutex> lock(fr->mutexPoseRel);
                        unique_lock<mutex> lock2(f2->mutexPoseRel);
                        fr->poseRel[f2] = Sim3((fr->getPose() * f2->getPose().inverse()).matrix());
                        f2->poseRel[fr] = Sim3((f2->getPose() * fr->getPose().inverse()).matrix());
                    }
                }
            }
        } else {
            // only record the reference and first frame and also update the keyframe poses in window
            {
                unique_lock<mutex> lock(frame->mutexPoseRel);
                frame->poseRel[refFrame] = Sim3((frame->getPose() * refFrame->getPose().inverse()).matrix());
                auto firstFrame = frames.front();
                frame->poseRel[firstFrame] = Sim3((frame->getPose() * firstFrame->getPose().inverse()).matrix());
            }

            // update the poses in window
            for (auto &fr: frames) {
                if (fr == frame) continue;
                for (auto rel: fr->poseRel) {
                    auto f2 = rel.first;
                    fr->poseRel[f2] = Sim3((fr->getPose() * f2->getPose().inverse()).matrix());
                }
            }
        }

        // visualization
        if (viewer)
            viewer->publishKeyframes(frames, false, Hcalib->mpCH);

        // =========================== Marginalize Frames =========================
        {
            unique_lock<mutex> lck(framesMutex);
            for (unsigned int i = 0; i < frames.size(); i++)
                if (frames[i]->frameHessian->flaggedForMarginalization) {
                    LOG(INFO) << "marg frame " << frames[i]->id << endl;
                    CHECK(frames[i] != coarseTracker->lastRef->frame);
                    marginalizeFrame(frames[i]);  //  边缘化不需要的帧
                    i = 0;
                }
        }

        // add current kf into map and detect loops
        globalMap->AddKeyFrame(fh->frame);
        if (setting_enableLoopClosing) {
            loopClosing->InsertKeyFrame(frame);
        }
        LOG(INFO) << "make keyframe done" << endl;
    }

    // for multi-fisheye
    void FullSystem::makeKeyFrameforMF(shared_ptr<FrameHessian> fh)
    {
        // // test
        // vector<shared_ptr<FrameHessian >> fhs;
        // for (auto &f: frames) fhs.push_back(f->frameHessian);
        // coarseTracker_forNewKF->setCoarseTrackingRefforMF(fhs);   // 把优化完的最新帧设为参考帧, 并估计逆深度
        // coarseTracker_forNewKF->debugPlotIDepthMapforMF(&minIdJetVisTracker, &maxIdJetVisTracker, viewer);


        shared_ptr<Frame> frame = fh->frame;
        auto refFrame = frames.back();

        {
            unique_lock<mutex> crlock(shellPoseMutex);
            fh->setEvalPT_scaledforMF(fh->frame->getPose(), fh->frame->vaff_g2l);
            fh->frame->setPoseOpti(Sim3(fh->frame->getPose().matrix()));
        }

        LOG(INFO) << "frame " << fh->frame->id << " is marked as key frame, active keyframes: " << frames.size()
                  << endl;

        // trace new keyframe 利用当前帧对前面关键帧中的未成熟点进行逆深度更新
        traceNewCoarseforMF(fh);

        unique_lock<mutex> lock(mapMutex);

         // == flag frames to be marginalized  == //  标记后面需要边缘化（从活动窗口踢出）的帧
        flagFramesForMarginalizationforMF(fh);

        // add new frame to hessian struct
        {
            unique_lock<mutex> lck(framesMutex);
            fh->idx = frames.size();
            frames.push_back(fh->frame);       // 将当前帧加入到滑动窗口中
            fh->frame->kfId = fh->frameID = globalMap->NumFrames();
        }

        ef->insertFrameforMF(fh);   // 将当前帧加入到总的能量函数中EnergyFunctional（ef）
        setPrecalcValuesforMF();    // 计算一下该窗口中其他帧与当前帧之间的一些参数比如相对光度、距离 每添加一个关键帧都会运行这个来设置位姿, 设置位姿线性化点

        LOG(INFO) << "adding new residuals" << endl;
        int numFwdResAdde = 0;

        for (auto fht : frames)  // go through all active frames  遍历窗口中之前所有帧
        {
            shared_ptr<FrameHessian> &fh1 = fht->frameHessian;
            if (fh1 == fh)
                continue;
            for (auto feat: fht->features) 
            {    // 遍历窗口中之前所有帧的成熟点
                if (feat->status == Feature::FeatureStatus::VALID
                    && feat->point->status == Point::PointStatus::ACTIVE) 
                {

                    shared_ptr<PointHessian> ph = feat->point->mpPH;

                    // add new residuals into this point hessian
                    shared_ptr<PointFrameResidual> r(
                        new PointFrameResidual(ph, fh1, fh));    // residual from fh1 to fh

                    r->setState(ResState::IN);
                    ph->residuals.push_back(r);
                    ef->insertResidual(r);      //  构建它们和新的关键帧的点帧误差PointFrameResidual，加入到ef中

                    ph->lastResiduals[1] = ph->lastResiduals[0];    // 设置上上个残差
                    ph->lastResiduals[0] = pair<shared_ptr<PointFrameResidual>, ResState>(r, ResState::IN); // 当前的设置为上一个
                    numFwdResAdde++;
                }
            }
        }

        // =========================== Activate Points (& flag for marginalization). =========================
        activatePointsMTforMF();    // 激活窗口中之前所有帧中符合条件的未成熟点，将其加入到ef中, 将host帧中的点投影到其他帧中，然后进行优化    
        ef->makeIDX();

        // =========================== OPTIMIZE ALL =========================
        fh->frameEnergyTH = frames.back()->frameHessian->frameEnergyTH;
        LOG(INFO) << "call optimize on kf " << frame->kfId << endl;
        float rmse = optimizeforMF(setting_maxOptIterations);    // 对活动窗口中的所有变量进行优化
        LOG(INFO) << "optimize is done! rmse is :" << rmse << endl;

        // =========================== Figure Out if INITIALIZATION FAILED =========================
        int numKFs = globalMap->NumFrames();
        // if (numKFs <= 4) 
        // {
        //     if (numKFs == 2 && rmse > 20 * benchmark_initializerSlackFactor) 
        //     {
        //         LOG(WARNING) << "I THINK INITIALIZATINO FAILED! Resetting." << endl;
        //         LOG(INFO) << "rmse = " << rmse << endl;
        //         initFailed = true;
        //     }
        //     if (numKFs == 3 && rmse > 13 * benchmark_initializerSlackFactor) 
        //     {
        //         LOG(WARNING) << "I THINK INITIALIZATINO FAILED! Resetting." << endl;
        //         LOG(INFO) << "rmse = " << rmse << endl;
        //         initFailed = true;
        //     }
        //     if (numKFs == 4 && rmse > 9 * benchmark_initializerSlackFactor) 
        //     {
        //         LOG(WARNING) << "I THINK INITIALIZATINO FAILED! Resetting." << endl;
        //         LOG(INFO) << "rmse = " << rmse << endl;
        //         initFailed = true;
        //     }
        // }

        if(numKFs <= 4)
	    {
            if(numKFs==2 && rmse > 30*benchmark_initializerSlackFactor)
            {
                printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
                initFailed=true;
            }
            if(numKFs==3 && rmse > 25*benchmark_initializerSlackFactor)
            {
                printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
                initFailed=true;
            }
            if(numKFs==4 && rmse > 20*benchmark_initializerSlackFactor)
            {
                printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
                initFailed=true;
            }
	    }

        if (isLost)
            return;

        // =========================== REMOVE OUTLIER =========================
        removeOutliersforMF();     // 去除外点

        // eval new energy!
        Vec3 newEnergy = linearizeAllforMF(false);
        double newEnergyL = calcLEnergyforMF();
        double newEnergyM = calcMEnergyforMF();
        cout << "remove outlier!" << endl;
        printOptResforMF(newEnergy, newEnergyL, newEnergyM, 0, 0, frames.back()->frameHessian->aff_g2lforMF());

        // swap the coarse Tracker for new kf
        {
            unique_lock<mutex> crlock(coarseTrackerSwapMutex);
            // coarseTracker_forNewKF->makeK(Hcalib->mpCH);
            vector<shared_ptr<FrameHessian >> fhs;
            for (auto &f: frames) fhs.push_back(f->frameHessian);
            coarseTracker_forNewKF->setCoarseTrackingRefforMF(fhs);   // 把优化完的最新帧设为参考帧, 并估计逆深度

            coarseTracker_forNewKF->debugPlotIDepthMapforMF(&minIdJetVisTracker, &maxIdJetVisTracker, viewer);
        }

        // =========================== (Activate-)Marginalize Points =========================
        // traditional bundle adjustment when marging all points
        flagPointsForRemovalforMF();   
        ef->dropPointsF();
        //每次设置线性化点都会更新零空间
        getNullspacesforMF(
            ef->lastNullspaces_pose,
            ef->lastNullspaces_scale);

        ef->marginalizePointsFforMF();      // 边缘化不需要的点

        // =========================== add new Immature points & new residuals =========================
        makeNewTracesforMF(fh, 0);    // 在当前帧中提取未成熟点

        // record the relative poses, note we are building a covisibility graph in fact
        auto minandmax = std::minmax_element(frames.begin(), frames.end(), CmpFrameKFID());
        unsigned long minKFId = (*minandmax.first)->kfId;
        unsigned long maxKFId = (*minandmax.second)->kfId;

        if (setting_fastLoopClosing == false) {
            // record all active keyframes
            for (auto &fr : frames) {
                auto allKFs = globalMap->GetAllKFs();
                for (auto &f2: allKFs) {
                    if (f2->kfId > minKFId && f2->kfId < maxKFId && f2 != fr) {
                        unique_lock<mutex> lock(fr->mutexPoseRel);
                        unique_lock<mutex> lock2(f2->mutexPoseRel);
                        fr->poseRel[f2] = Sim3((fr->getPose() * f2->getPose().inverse()).matrix());
                        f2->poseRel[fr] = Sim3((f2->getPose() * fr->getPose().inverse()).matrix());
                    }
                }
            }
        } else {
            // only record the reference and first frame and also update the keyframe poses in window
            {
                unique_lock<mutex> lock(frame->mutexPoseRel);
                frame->poseRel[refFrame] = Sim3((frame->getPose() * refFrame->getPose().inverse()).matrix());
                auto firstFrame = frames.front();
                frame->poseRel[firstFrame] = Sim3((frame->getPose() * firstFrame->getPose().inverse()).matrix());
            }

            // update the poses in window
            for (auto &fr: frames) {
                if (fr == frame) continue;
                for (auto rel: fr->poseRel) {
                    auto f2 = rel.first;
                    fr->poseRel[f2] = Sim3((fr->getPose() * f2->getPose().inverse()).matrix());
                }
            }
        }

        LOG(INFO) << "T wc (keyframe) :" << endl << fh->get_worldToCam_evalPT().matrix().inverse() <<endl;

        //visualization
        if (viewer)
            viewer->publishKeyframesforMF(frames, false, UMF);

        // =========================== Marginalize Frames =========================
        {
            unique_lock<mutex> lck(framesMutex);
            for (unsigned int i = 0; i < frames.size(); i++)
                if (frames[i]->frameHessian->flaggedForMarginalization) 
                {
                    LOG(INFO) << "marg frame " << frames[i]->id << endl;
                    CHECK(frames[i] != coarseTracker->lastRef->frame);
                    marginalizeFrameforMF(frames[i]);  //  边缘化不需要的帧
                    i = 0;
                }
        }

        // add current kf into map and detect loops
        globalMap->AddKeyFrame(fh->frame);
        // if (setting_enableLoopClosing) {
        //     loopClosing->InsertKeyFrame(frame);
        // }
        LOG(INFO) << "make keyframe done" << endl;
    }
    // 如果当前帧被认为是非关键帧，那么该帧就用来对活动窗口中所有的关键帧中还未成熟的点进行逆深度更新。基本原理是沿着极线进行搜索
    void FullSystem::makeNonKeyFrame(shared_ptr<FrameHessian> &fh) {
        {
            unique_lock<mutex> crlock(shellPoseMutex);
            fh->setEvalPT_scaled(fh->frame->getPose(), fh->frame->aff_g2l);
            LOG(INFO) << "T wc (Non keyframe) :" << endl << fh->get_worldToCam_evalPT().matrix().inverse() <<endl;
        }
        traceNewCoarse(fh);
        fh->frame->ReleaseAll();  // no longer needs it
    }

    // for multi-fisheye
    // 如果当前帧被认为是非关键帧，那么该帧就用来对活动窗口中所有的关键帧中还未成熟的点进行逆深度更新。基本原理是沿着极线进行搜索
    void FullSystem::makeNonKeyFrameforMF(shared_ptr<FrameHessian> &fh) 
    {
        {
            unique_lock<mutex> crlock(shellPoseMutex);
            fh->setEvalPT_scaledforMF(fh->frame->getPose(), fh->frame->vaff_g2l);
        }
        traceNewCoarseforMF(fh); 
        fh->frame->ReleaseAll();  // no longer needs it
    }

    void FullSystem::marginalizeFrame(shared_ptr<Frame> &frame) {

        // marginalize or remove all this frames points
        ef->marginalizeFrame(frame->frameHessian);

        // drop all observations of existing points in that frame
        for (shared_ptr<Frame> &fr: frames) {
            if (fr == frame)
                continue;
            for (auto feat: fr->features) {
                if (feat->status == Feature::FeatureStatus::VALID &&
                    feat->point->status == Point::PointStatus::ACTIVE) {

                    shared_ptr<PointHessian> ph = feat->point->mpPH;
                    // remove the residuals projected into this frame
                    size_t n = ph->residuals.size();
                    for (size_t i = 0; i < n; i++) {
                        auto r = ph->residuals[i];
                        if (r->target.lock() == frame->frameHessian) {
                            if (ph->lastResiduals[0].first == r)
                                ph->lastResiduals[0].first = nullptr;
                            else if (ph->lastResiduals[1].first == r)
                                ph->lastResiduals[1].first = nullptr;
                            ef->dropResidual(r);
                            i--;
                            n--;
                        }
                    }
                }
            }
        }

        // remove this frame from recorded frames
        frame->ReleaseAll();    // release all things in this frame
        deleteOutOrder<shared_ptr<Frame>>
            (frames, frame);

        // reset the optimization idx
        for (unsigned int i = 0; i < frames.size(); i++)
            frames[i]->frameHessian->idx = i;

        setPrecalcValues();
        ef->setAdjointsF(Hcalib->mpCH);
    }

    // for multi-fisheye
    void FullSystem::marginalizeFrameforMF(shared_ptr<Frame> &frame) {

        // marginalize or remove all this frames points
        ef->marginalizeFrameforMF(frame->frameHessian);

        // drop all observations of existing points in that frame
        for (shared_ptr<Frame> &fr: frames) {
            if (fr == frame)
                continue;
            for (auto feat: fr->features) {
                if (feat->status == Feature::FeatureStatus::VALID &&
                    feat->point->status == Point::PointStatus::ACTIVE) {

                    shared_ptr<PointHessian> ph = feat->point->mpPH;
                    // remove the residuals projected into this frame
                    size_t n = ph->residuals.size();
                    for (size_t i = 0; i < n; i++) {
                        auto r = ph->residuals[i];
                        if (r->target.lock() == frame->frameHessian) {
                            if (ph->lastResiduals[0].first == r)
                                ph->lastResiduals[0].first = nullptr;
                            else if (ph->lastResiduals[1].first == r)
                                ph->lastResiduals[1].first = nullptr;
                            ef->dropResidual(r);
                            i--;
                            n--;
                        }
                    }
                }
            }
        }

        // remove this frame from recorded frames
        frame->ReleaseAll();    // release all things in this frame
        deleteOutOrder<shared_ptr<Frame>>
            (frames, frame);

        // reset the optimization idx
        for (unsigned int i = 0; i < frames.size(); i++)
            frames[i]->frameHessian->idx = i;

        setPrecalcValuesforMF();
        ef->setAdjointsFforMF();
    }

    void FullSystem::flagFramesForMarginalization(shared_ptr<FrameHessian> &newFH) {

        if (setting_minFrameAge > setting_maxFrames) {
            for (size_t i = setting_maxFrames; i < frames.size(); i++) {
                shared_ptr<FrameHessian> &fh = frames[i - setting_maxFrames]->frameHessian;
                LOG(INFO) << "frame " << fh->frame->kfId << " is set as marged" << endl;
                fh->flaggedForMarginalization = true;
            }
            return;
        }

        int flagged = 0;

        // marginalize all frames that have not enough points.
        for (int i = 0; i < (int) frames.size(); i++) {

            shared_ptr<FrameHessian> &fh = frames[i]->frameHessian;
            int in = 0, out = 0;
            for (auto &feat: frames[i]->features) {
                if (feat->status == Feature::FeatureStatus::IMMATURE) {
                    in++;
                    continue;
                }

                shared_ptr<Point> p = feat->point;
                if (p && p->status == Point::PointStatus::ACTIVE)
                    in++;
                else
                    out++;
            }

            Vec2 refToFh = AffLight::fromToVecExposure(frames.back()->frameHessian->ab_exposure, fh->ab_exposure,
                                                       frames.back()->frameHessian->aff_g2l(), fh->aff_g2l());

            // some kind of marginlization conditions   边缘化一帧中活跃的点少于５% 或者 最新的帧相比光度参数变化剧烈
            if ((in < setting_minPointsRemaining * (in + out) ||
                 fabs(logf((float) refToFh[0])) > setting_maxLogAffFacInWindow)
                && ((int) frames.size()) - flagged > setting_minFrames) {
                LOG(INFO) << "frame " << fh->frame->kfId << " is set as marged" << endl;
                fh->flaggedForMarginalization = true;
                flagged++;
            }
        }

        // still too much, marginalize one  如果过程１没有找到需要边缘化的帧，则从全部帧中找到除最近的两帧外离当前帧最远的一帧
        if ((int) frames.size() - flagged >= setting_maxFrames) {
            double smallestScore = 1;
            shared_ptr<Frame> toMarginalize = nullptr;
            shared_ptr<Frame> latest = frames.back();

            for (auto &fr: frames) {
                if (fr->frameHessian->frameID > latest->frameHessian->frameID - setting_minFrameAge ||
                    fr->frameHessian->frameID == 0)
                    continue;
                double distScore = 0;
                for (FrameFramePrecalc &ffh: fr->frameHessian->targetPrecalc) {
                    if (ffh.target.lock()->frameID > latest->frameHessian->frameID - setting_minFrameAge + 1 ||
                        ffh.target.lock() == ffh.host.lock())
                        continue;

                    distScore += 1 / (1e-5 + ffh.distanceLL);
                }
                distScore *= -sqrtf(fr->frameHessian->targetPrecalc.back().distanceLL);

                if (distScore < smallestScore) {
                    smallestScore = distScore;
                    toMarginalize = fr;
                }
            }

            if (toMarginalize) {
                toMarginalize->frameHessian->flaggedForMarginalization = true;
                LOG(INFO) << "frame " << toMarginalize->kfId << " is set as marged" << endl;
                flagged++;
            }
        }
    }

    // for multi-fisheye
    void FullSystem::flagFramesForMarginalizationforMF(shared_ptr<FrameHessian> &newFH)
    {
        if (setting_minFrameAge > setting_maxFrames) 
        {
            for (size_t i = setting_maxFrames; i < frames.size(); i++) {
                shared_ptr<FrameHessian> &fh = frames[i - setting_maxFrames]->frameHessian;
                LOG(INFO) << "frame " << fh->frame->kfId << " is set as marged" << endl;
                fh->flaggedForMarginalization = true;
            }
            return;
        }

        int flagged = 0;

        // marginalize all frames that have not enough points.
        for (int i = 0; i < (int) frames.size(); i++) 
        {

            shared_ptr<FrameHessian> &fh = frames[i]->frameHessian;
            int in = 0, out = 0;
            for (auto &feat: frames[i]->features) 
            {
                if (feat->status == Feature::FeatureStatus::IMMATURE) 
                {
                    in++;
                    continue;
                }

                shared_ptr<Point> p = feat->point;
                if (p && p->status == Point::PointStatus::ACTIVE)
                    in++;
                else
                    out++;
            }

            vector<vector<Vec2>> vvrefToFh;
            vector<double> va;
            vvrefToFh.resize(UMF->camNums);
            for(int n1 = 0; n1 < UMF->camNums; n1++)
            {
                vvrefToFh[n1].resize(UMF->camNums);
                for(int n2 = 0; n2 < UMF->camNums; n2++)
                {
                    vvrefToFh[n1][n2] = AffLight::fromToVecExposure(frames.back()->frameHessian->ab_exposure, fh->ab_exposure,
                                                frames.back()->frameHessian->aff_g2lforMF()[n1], fh->aff_g2lforMF()[n2]);
                    va.emplace_back(vvrefToFh[n1][n2][0]);
                }
            }
            float maxa = *max_element(va.begin(), va.end());

            // some kind of marginlization conditions   边缘化一帧中活跃的点(留下的点：去除边缘化+删除)少于５% 或者 最新的帧相比光度参数变化剧烈
            if ((in < setting_minPointsRemaining * (in + out) ||
                 fabs(logf(maxa)) > setting_maxLogAffFacInWindow)
                && ((int) frames.size()) - flagged > setting_minFrames) 
            {
                LOG(INFO) << "frame " << fh->frame->kfId << " is set as marged" << endl;
                fh->flaggedForMarginalization = true;
                flagged++;
            }
        }

        // still too much, marginalize one  如果过程１没有找到需要边缘化的帧，则从全部帧中找到除最近的两帧外离当前帧最远的一帧
        if ((int) frames.size() - flagged >= setting_maxFrames) {
            double smallestScore = 1;
            shared_ptr<Frame> toMarginalize = nullptr;
            shared_ptr<Frame> latest = frames.back();

            for (auto &fr: frames) 
            {
                if (fr->frameHessian->frameID > latest->frameHessian->frameID - setting_minFrameAge ||
                    fr->frameHessian->frameID == 0)
                    continue;
                double distScore = 0;
                for (FrameFramePrecalc &ffh: fr->frameHessian->targetPrecalc) 
                {
                    if (ffh.target.lock()->frameID > latest->frameHessian->frameID - setting_minFrameAge + 1 ||
                        ffh.target.lock() == ffh.host.lock())
                        continue;

                    distScore += 1 / (1e-5 + ffh.distanceLL);
                }
                distScore *= -sqrtf(fr->frameHessian->targetPrecalc.back().distanceLL);

                if (distScore < smallestScore) 
                {
                    smallestScore = distScore;
                    toMarginalize = fr;
                }
            }

            if (toMarginalize) 
            {
                toMarginalize->frameHessian->flaggedForMarginalization = true;
                LOG(INFO) << "frame " << toMarginalize->kfId << " is set as marged" << endl;
                flagged++;
            }
        }
    }

    float FullSystem::optimize(int mnumOptIts) {

        if (frames.size() < 2)
            return 0;
        if (frames.size() < 3)
            mnumOptIts = 20;
        if (frames.size() < 4)
            mnumOptIts = 15;

        // get statistics and active residuals.
        activeResiduals.clear();
        int numPoints = 0;
        int numLRes = 0;
        for (shared_ptr<Frame> &fr : frames) {
            for (auto &feat: fr->features) {
                shared_ptr<Point> p = feat->point;
                if (feat->status == Feature::FeatureStatus::VALID && p
                    && p->status == Point::PointStatus::ACTIVE) {
                    auto ph = p->mpPH;
                    for (auto &r : ph->residuals) {
                        if (!r->isLinearized) {
                            activeResiduals.push_back(r);
                            r->resetOOB();
                        } else {
                            numLRes++;
                        }
                    }
                }
                numPoints++;
            }
        }

        LOG(INFO) << "active residuals: " << activeResiduals.size() << endl;
        // 计算相关导数
        Vec3 lastEnergy = linearizeAll(false);
        double lastEnergyL = calcLEnergy();
        double lastEnergyM = calcMEnergy();

        // apply res
        if (multiThreading)
            threadReduce.reduce(bind(&FullSystem::applyRes_Reductor, this, true, _1, _2, _3, _4), 0,
                                activeResiduals.size(), 50);
        else
            applyRes_Reductor(true, 0, activeResiduals.size(), 0, 0);

        printOptRes(lastEnergy, lastEnergyL, lastEnergyM, 0, 0, frames.back()->frameHessian->aff_g2l().a,
                    frames.back()->frameHessian->aff_g2l().b);

        double lambda = 1e-1;
        float stepsize = 1;
        VecX previousX = VecX::Constant(CPARS + 8 * frames.size(), NAN);
        // 相关导数计算完成，进行迭代优化
        for (int iteration = 0; iteration < mnumOptIts; iteration++) {
            // solve!  保存当前的状态
            backupState(iteration != 0);

            solveSystem(iteration, lambda);
            double incDirChange = (1e-20 + previousX.dot(ef->lastX)) / (1e-20 + previousX.norm() * ef->lastX.norm());
            previousX = ef->lastX;

            if (std::isfinite(incDirChange) && (setting_solverMode & SOLVER_STEPMOMENTUM)) {
                float newStepsize = exp(incDirChange * 1.4);
                if (incDirChange < 0 && stepsize > 1) stepsize = 1;

                stepsize = sqrtf(sqrtf(newStepsize * stepsize * stepsize * stepsize));
                if (stepsize > 2) stepsize = 2;
                if (stepsize < 0.25) stepsize = 0.25;
            }

            bool canbreak = doStepFromBackup(stepsize, stepsize, stepsize, stepsize, stepsize);

            // eval new energy!
            Vec3 newEnergy = linearizeAll(false);
            double newEnergyL = calcLEnergy();
            double newEnergyM = calcMEnergy();

            printOptRes(newEnergy, newEnergyL, newEnergyM, 0, 0, frames.back()->frameHessian->aff_g2l().a,
                        frames.back()->frameHessian->aff_g2l().b);

            // control the lambda in LM
            if (setting_forceAceptStep || (newEnergy[0] + newEnergy[1] + newEnergyL + newEnergyM <
                                           lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM)) {

                // energy is decreasing
                if (multiThreading)
                    threadReduce.reduce(bind(&FullSystem::applyRes_Reductor, this, true, _1, _2, _3, _4), 0,
                                        activeResiduals.size(), 50);
                else
                    applyRes_Reductor(true, 0, activeResiduals.size(), 0, 0);

                lastEnergy = newEnergy;
                lastEnergyL = newEnergyL;
                lastEnergyM = newEnergyM;

                lambda *= 0.25;
            } else {
                // energy increses, reload the backup state and increase lambda
                loadSateBackup();
                lastEnergy = linearizeAll(false);
                lastEnergyL = calcLEnergy();
                lastEnergyM = calcMEnergy();
                lambda *= 1e2;
            }

            if (canbreak && iteration >= setting_minOptIterations)
                break;
        }

        Vec10 newStateZero = Vec10::Zero();
        newStateZero.segment<2>(6) = frames.back()->frameHessian->get_state().segment<2>(6);

        frames.back()->frameHessian->setEvalPT(frames.back()->frameHessian->PRE_worldToCam, newStateZero);

        EFDeltaValid = false;
        EFAdjointsValid = false;
        ef->setAdjointsF(Hcalib->mpCH);
        setPrecalcValues();

        lastEnergy = linearizeAll(true);    // fix all the linearizations

        if (!std::isfinite((double) lastEnergy[0]) || !std::isfinite((double) lastEnergy[1]) ||
            !std::isfinite((double) lastEnergy[2])) {
            LOG(WARNING) << "KF Tracking failed: LOST!";
            isLost = true;
        }

        // set the estimated pose into frame
        {
            unique_lock<mutex> crlock(shellPoseMutex);
            for (auto fr: frames) {
                fr->setPose(fr->frameHessian->PRE_camToWorld.inverse());
                fr->aff_g2l = fr->frameHessian->aff_g2l();
            }
        }

        return sqrtf((float) (lastEnergy[0] / (patternNum * ef->resInA)));
    }

    // for multi-fisheye
    float FullSystem::optimizeforMF(int mnumOptIts)
    {
        if (frames.size() < 2)
            return 0;
        if (frames.size() < 3)
            mnumOptIts = 20;
        if (frames.size() < 4)
            mnumOptIts = 15;

        // get statistics and active residuals.找出未线性化(边缘化)的残差, 加入activeResiduals
        activeResiduals.clear();
        int numPoints = 0;
        int numLRes = 0;
        for (shared_ptr<Frame> &fr : frames) 
        {
            for (auto &feat: fr->features) 
            {
                shared_ptr<Point> p = feat->point;
                if (feat->status == Feature::FeatureStatus::VALID && p
                    && p->status == Point::PointStatus::ACTIVE) 
                {
                    auto ph = p->mpPH;
                    for (auto &r : ph->residuals)   
                    {
                        if (!r->isLinearized)       // 没有求线性误差
                        {
                            activeResiduals.push_back(r);       // 新加入的残差
                            r->resetOOB();                      // residual状态重置
                        } 
                        else 
                        {
                            numLRes++;          //已经线性化过的计数
                        }
                    }
                }
                numPoints++;
            }
        }

        LOG(INFO) << "active residuals: " << activeResiduals.size() << endl;
        // 线性化activeResiduals的残差, 计算边缘化的能量值 (然而这里都设成0了)
	    // 线性化, 参数: [true是进行固定线性化, 并去掉不好的残差] [false不进行固定线性化]
        Vec3 lastEnergy = linearizeAllforMF(false);
        double lastEnergyL = calcLEnergyforMF();     // islinearized的量的能量  
        double lastEnergyM = calcMEnergyforMF();     // HM部分的能量  待修改

        // apply res
        if (multiThreading)
            threadReduce.reduce(bind(&FullSystem::applyRes_Reductor, this, true, _1, _2, _3, _4), 0,
                                activeResiduals.size(), 50);
        else
            applyRes_Reductor(true, 0, activeResiduals.size(), 0, 0);

        printOptResforMF(lastEnergy, lastEnergyL, lastEnergyM, 0, 0, frames.back()->frameHessian->aff_g2lforMF());

        double lambda = 1e-1;
        float stepsize = 1;
        int camnum = UMF->camNums;
        VecX previousX = VecX::Constant(6 * frames.size() + frames.size() * 2 * camnum , NAN);
        
        // 相关导数计算完成，进行迭代优化
        for (int iteration = 0; iteration < mnumOptIts; iteration++)
        {
            // solve!  保存当前的状态
            backupStateforMF(iteration != 0);

            solveSystemforMF(iteration, lambda);

            double incDirChange = (1e-20 + previousX.dot(ef->lastX)) / (1e-20 + previousX.norm() * ef->lastX.norm());
            previousX = ef->lastX;

            // cout << "ef->lastX" << ef->lastX << endl;

            if (std::isfinite(incDirChange) && (setting_solverMode & SOLVER_STEPMOMENTUM)) 
            {
                float newStepsize = exp(incDirChange * 1.4);
                if (incDirChange < 0 && stepsize > 1) stepsize = 1;

                stepsize = sqrtf(sqrtf(newStepsize * stepsize * stepsize * stepsize));
                if (stepsize > 2) stepsize = 2;
                if (stepsize < 0.25) stepsize = 0.25;
            }
            //* 更新变量, 判断是否停止
            bool canbreak = doStepFromBackupforMF(stepsize, stepsize, stepsize, stepsize);
        
            // eval new energy!
            Vec3 newEnergy = linearizeAllforMF(false);
            double newEnergyL = calcLEnergyforMF();
            double newEnergyM = calcMEnergyforMF();

            printOptResforMF(newEnergy, newEnergyL, newEnergyM, 0, 0, frames.back()->frameHessian->aff_g2lforMF());
            //setting_forceAceptStep = false;
            // control the lambda in LM
            if (setting_forceAceptStep || (newEnergy[0] + newEnergy[1] + newEnergyL + newEnergyM <
                                           lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM)) {

                // energy is decreasing
                if (multiThreading)
                    threadReduce.reduce(bind(&FullSystem::applyRes_Reductor, this, true, _1, _2, _3, _4), 0,
                                        activeResiduals.size(), 50);
                else
                    applyRes_Reductor(true, 0, activeResiduals.size(), 0, 0);

                lastEnergy = newEnergy;
                lastEnergyL = newEnergyL;
                lastEnergyM = newEnergyM;

                lambda *= 0.25;
            } else {
                // energy increses, reload the backup state and increase lambda
                loadSateBackupforMF();
                lastEnergy = linearizeAllforMF(false);
                lastEnergyL = calcLEnergyforMF();
                lastEnergyM = calcMEnergyforMF();
                lambda *= 1e2;
            }

            if (canbreak && iteration >= setting_minOptIterations)
                break;
        }

        Vec16 newStateZero = Vec16::Zero();
        newStateZero.segment<10>(6) = frames.back()->frameHessian->get_stateforMF().segment<10>(6);

        frames.back()->frameHessian->setEvalPTforMF(frames.back()->frameHessian->PRE_worldToCam, newStateZero);

        EFDeltaValid = false;
        EFAdjointsValid = false;
        ef->setAdjointsFforMF();
        setPrecalcValuesforMF();

        lastEnergy = linearizeAllforMF(true);    // fix all the linearizations

        if (!std::isfinite((double) lastEnergy[0]) || !std::isfinite((double) lastEnergy[1]) ||
            !std::isfinite((double) lastEnergy[2])) {
            LOG(WARNING) << "KF Tracking failed: LOST!";
            isLost = true;
        }

        // set the estimated pose into frame
        {
            unique_lock<mutex> crlock(shellPoseMutex);
            for (auto fr: frames) {
                fr->setPose(fr->frameHessian->PRE_camToWorld.inverse());
                fr->vaff_g2l = fr->frameHessian->aff_g2lforMF();
            }
        }
        // // output points
        // string path = "/home/jiangfan/桌面/pan_dso_calib/pano_tracking/pcop.txt";
        // ofstream outp;
        // outp.open(path.c_str());
        // int idp = 0;
        // int idf = 0;
        // for (auto &fr:frames) 
        //     {
        //         for (auto feat: fr->features) 
        //         {
        //             if (feat->status == Feature::FeatureStatus::VALID
        //                 && feat->point && feat->point->status == Point::PointStatus::ACTIVE) 
        //             {
        //                 auto ph = feat->point->mpPH;
        //                 outp << idf << " " << ph->xs << " " << ph->ys << " " << ph->zs << " " << ph->idepth << endl;
        //                 idp ++;
        //             }
        //         }
        //         idf ++;
        //     }
        // outp.close();

        return sqrtf((float) (lastEnergy[0] / (patternNum * ef->resInA)));
    }

    void FullSystem::setGammaFunction(float *BInv) {

        if (BInv == nullptr)
            return;

        // copy BInv.
        memcpy(Hcalib->mpCH->Binv, BInv, sizeof(float) * 256);

        // invert.
        for (int i = 0; i < 255; i++) {

            // find val, such that Binv[val] = i.
            // I dont care about speed for this, so do it the stupid way.

            for (int s = 1; s < 255; s++) {
                if (BInv[s] <= i && BInv[s + 1] >= i) {
                    Hcalib->mpCH->B[i] = s + (i - BInv[s]) / (BInv[s + 1] - BInv[s]);
                    break;
                }
            }
        }

        Hcalib->mpCH->B[0] = 0;
        Hcalib->mpCH->B[255] = 255;
    }

    shared_ptr<PointHessian>
    FullSystem::optimizeImmaturePoint(shared_ptr<internal::ImmaturePoint> point, int minObs,
                                      vector<shared_ptr<ImmaturePointTemporaryResidual>> &residuals) {
        int nres = 0;
        shared_ptr<Frame> hostFrame = point->feature->host.lock();
        assert(hostFrame);  // the feature should have a host frame

        for (auto fr: frames) {
            if (fr != hostFrame) {
                residuals[nres]->state_NewEnergy = residuals[nres]->state_energy = 0;
                residuals[nres]->state_NewState = ResState::OUTLIER;
                residuals[nres]->state_state = ResState::IN;
                residuals[nres]->target = fr->frameHessian;
                nres++;
            }
        }
        assert(nres == frames.size() - 1);

        float lastEnergy = 0;
        float lastHdd = 0;
        float lastbd = 0;
        float currentIdepth = (point->idepth_max + point->idepth_min) * 0.5f;

        for (int i = 0; i < nres; i++) {
            lastEnergy += point->linearizeResidual(Hcalib->mpCH, 1000, residuals[i], lastHdd, lastbd, currentIdepth);
            residuals[i]->state_state = residuals[i]->state_NewState;
            residuals[i]->state_energy = residuals[i]->state_NewEnergy;
        }

        if (!std::isfinite(lastEnergy) || lastHdd < setting_minIdepthH_act) {
            return 0;
        }

        // do LM iteration for this immature point
        float lambda = 0.1;
        for (int iteration = 0; iteration < setting_GNItsOnPointActivation; iteration++) {
            float H = lastHdd;
            H *= 1 + lambda;
            float step = (1.0 / H) * lastbd;
            float newIdepth = currentIdepth - step;

            float newHdd = 0;
            float newbd = 0;
            float newEnergy = 0;
            for (int i = 0; i < nres; i++) {
                // compute the energy in other frames
                newEnergy += point->linearizeResidual(Hcalib->mpCH, 1, residuals[i], newHdd, newbd, newIdepth);
            }

            if (!std::isfinite(lastEnergy) || newHdd < setting_minIdepthH_act) {
                return 0;
            }

            if (newEnergy < lastEnergy) {
                currentIdepth = newIdepth;
                lastHdd = newHdd;
                lastbd = newbd;
                lastEnergy = newEnergy;
                for (int i = 0; i < nres; i++) {
                    residuals[i]->state_state = residuals[i]->state_NewState;
                    residuals[i]->state_energy = residuals[i]->state_NewEnergy;
                }
                lambda *= 0.5;
            } else {
                lambda *= 5;
            }

            if (fabsf(step) < 0.0001 * currentIdepth)
                break;
        }

        if (!std::isfinite(currentIdepth)) {
            return nullptr;
        }

        int numGoodRes = 0;
        for (int i = 0; i < nres; i++)
            if (residuals[i]->state_state == ResState::IN)
                numGoodRes++;

        if (numGoodRes < minObs) {
            // an outlier
            return nullptr;
        }

        point->feature->CreateFromImmature();    // create a point from immature feature
        shared_ptr<PointHessian> p = point->feature->point->mpPH;

        // set residual status in new map point
        p->lastResiduals[0].first = nullptr;
        p->lastResiduals[0].second = ResState::OOB;
        p->lastResiduals[1].first = nullptr;
        p->lastResiduals[1].second = ResState::OOB;

        p->setIdepthZero(currentIdepth);
        p->setIdepth(currentIdepth);

        // move the immature point residuals into the new map point
        for (int i = 0; i < nres; i++)
            if (residuals[i]->state_state == ResState::IN) {
                shared_ptr<FrameHessian> host = point->feature->host.lock()->frameHessian;
                shared_ptr<FrameHessian> target = residuals[i]->target.lock();
                shared_ptr<PointFrameResidual> r(new PointFrameResidual(p, host, target));

                r->state_NewEnergy = r->state_energy = 0;
                r->state_NewState = ResState::OUTLIER;
                r->setState(ResState::IN);
                p->residuals.push_back(r);

                if (target == frames.back()->frameHessian) {
                    p->lastResiduals[0].first = r;
                    p->lastResiduals[0].second = ResState::IN;
                } else if (target == (frames.size() < 2 ? nullptr : frames[frames.size() - 2]->frameHessian)) {
                    p->lastResiduals[1].first = r;
                    p->lastResiduals[1].second = ResState::IN;
                }
            }
        return p;
    }

    // for multi-fisheye 优化未成熟点逆深度, 并创建成PointHessian
    shared_ptr<PointHessian>
    FullSystem::optimizeImmaturePointforMF(shared_ptr<internal::ImmaturePoint> point, int minObs,
                                      vector<shared_ptr<ImmaturePointTemporaryResidual>> &residuals)
    {
        // 初始化和其它关键帧的res(点在其它关键帧上投影)  每个点都从host投影到其他帧上
        int nres = 0;
        shared_ptr<Frame> hostFrame = point->feature->host.lock();
        assert(hostFrame);  // the feature should have a host frame
        // 设置ImmaturePointTemporaryResidual
        for (auto fr: frames) 
        {
            if (fr != hostFrame) {
                residuals[nres]->state_NewEnergy = residuals[nres]->state_energy = 0;
                residuals[nres]->state_NewState = ResState::OUTLIER;
                residuals[nres]->state_state = ResState::IN;
                residuals[nres]->target = fr->frameHessian;
                nres++; // 观测数
            }
        }

        assert(nres == frames.size() - 1);

        float lastEnergy = 0;
        float lastHdd = 0;
        float lastbd = 0;
        float currentIdepth = (point->idepth_max + point->idepth_min) * 0.5f;


        //  使用类LM(GN)的方法来优化逆深度(而不是使用三角化)哪种更好
        for (int i = 0; i < nres; i++) 
        {
            lastEnergy += point->linearizeResidualforMF(1000, residuals[i], lastHdd, lastbd, currentIdepth);
            residuals[i]->state_state = residuals[i]->state_NewState;
            residuals[i]->state_energy = residuals[i]->state_NewEnergy;
        }

        if (!std::isfinite(lastEnergy) || lastHdd < setting_minIdepthH_act) {
            return 0;
        }

        float lambda = 0.1;
        for (int iteration = 0; iteration < setting_GNItsOnPointActivation; iteration++) 
        {
            float H = lastHdd;
            H *= 1 + lambda;
            float step = (1.0 / H) * lastbd;
            float newIdepth = currentIdepth - step;

            float newHdd = 0;
            float newbd = 0;
            float newEnergy = 0;
            for (int i = 0; i < nres; i++) 
            {
                // compute the energy in other frames
                newEnergy += point->linearizeResidualforMF(1, residuals[i], newHdd, newbd, newIdepth);
            }

            if (!std::isfinite(lastEnergy) || newHdd < setting_minIdepthH_act) {
                return 0;
            }

            if (newEnergy < lastEnergy) {
                currentIdepth = newIdepth;
                lastHdd = newHdd;
                lastbd = newbd;
                lastEnergy = newEnergy;
                for (int i = 0; i < nres; i++) {
                    residuals[i]->state_state = residuals[i]->state_NewState;
                    residuals[i]->state_energy = residuals[i]->state_NewEnergy;
                }
                lambda *= 0.5;
            } else {
                lambda *= 5;
            }

            if (fabsf(step) < 0.0001 * currentIdepth)
                break;
        }

        if (!std::isfinite(currentIdepth)) {
            return nullptr;
        }

        int numGoodRes = 0;
        for (int i = 0; i < nres; i++)
            if (residuals[i]->state_state == ResState::IN)
                numGoodRes++;

        if (numGoodRes < minObs) {
            // an outlier
            return nullptr;
        }

        point->feature->CreateFromImmature();    // create a point from immature feature
        shared_ptr<PointHessian> p = point->feature->point->mpPH;

        // set residual status in new map point
        p->lastResiduals[0].first = nullptr;
        p->lastResiduals[0].second = ResState::OOB;
        p->lastResiduals[1].first = nullptr;
        p->lastResiduals[1].second = ResState::OOB;

        p->setIdepthZero(currentIdepth);
        p->setIdepth(currentIdepth);

        // move the immature point residuals into the new map point
        for (int i = 0; i < nres; i++)
            if (residuals[i]->state_state == ResState::IN) {
                shared_ptr<FrameHessian> host = point->feature->host.lock()->frameHessian;
                shared_ptr<FrameHessian> target = residuals[i]->target.lock();
                shared_ptr<PointFrameResidual> r(new PointFrameResidual(p, host, target));

                r->state_NewEnergy = r->state_energy = 0;
                r->state_NewState = ResState::OUTLIER;
                r->setState(ResState::IN);
                p->residuals.push_back(r);

                if (target == frames.back()->frameHessian) {
                    p->lastResiduals[0].first = r;
                    p->lastResiduals[0].second = ResState::IN;
                } else if (target == (frames.size() < 2 ? nullptr : frames[frames.size() - 2]->frameHessian)) {
                    p->lastResiduals[1].first = r;
                    p->lastResiduals[1].second = ResState::IN;
                }
            }
        return p;
    }

    void FullSystem::traceNewCoarse(shared_ptr<FrameHessian> fh) {

        unique_lock<mutex> lock(mapMutex);

        int trace_total = 0, trace_good = 0, trace_oob = 0, trace_out = 0, trace_skip = 0, trace_badcondition = 0, trace_uninitialized = 0;

        Mat33f K = Mat33f::Identity();
        K(0, 0) = Hcalib->mpCH->fxl();
        K(1, 1) = Hcalib->mpCH->fyl();
        K(0, 2) = Hcalib->mpCH->cxl();
        K(1, 2) = Hcalib->mpCH->cyl();

        for (shared_ptr<Frame> fr: frames) {  // 遍历 窗口内的所有关键帧
            shared_ptr<FrameHessian> host = fr->frameHessian;

            SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
            Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
            Vec3f Kt = K * hostToNew.translation().cast<float>();

            Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure, host->aff_g2l(),
                                                    fh->aff_g2l()).cast<float>();

            for (auto feat: fr->features) {
                if (feat->status == Feature::FeatureStatus::IMMATURE && feat->ip) {
                    // update the immature points
                    shared_ptr<ImmaturePoint> ph = feat->ip;
                    ph->traceOn(fh, KRKi, Kt, aff, Hcalib->mpCH);

                    if (ph->lastTraceStatus == ImmaturePointStatus::IPS_GOOD) trace_good++;
                    if (ph->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION) trace_badcondition++;
                    if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OOB) trace_oob++;
                    if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER) trace_out++;
                    if (ph->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED) trace_skip++;
                    if (ph->lastTraceStatus == ImmaturePointStatus::IPS_UNINITIALIZED) trace_uninitialized++;
                    trace_total++;
                }
            }
        }
    }

    // for multi-fisheye
    void FullSystem::traceNewCoarseforMF(shared_ptr<FrameHessian> fh)
    {
        unique_lock<mutex> lock(mapMutex);

        int trace_total = 0, trace_good = 0, trace_oob = 0, trace_out = 0, trace_skip = 0, trace_badcondition = 0, trace_uninitialized = 0;

        for(shared_ptr<Frame> fr : frames)
        {
            shared_ptr<FrameHessian> host = fr->frameHessian;

            SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
            Mat33f R = hostToNew.rotationMatrix().cast<float>();
            Vec3f t = hostToNew.translation().cast<float>();

            vector<vector<Vec2f>> vvaff;
            vvaff.resize(UMF->camNums);
            for(int n1 = 0; n1 < UMF->camNums; n1++)
            {
                vvaff[n1].resize(UMF->camNums);
                for(int n2 = 0; n2 < UMF->camNums; n2++)
                {
                    vvaff[n1][n2] = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure, host->aff_g2lforMF()[n1],
                                                    fh->aff_g2lforMF()[n2]).cast<float>();
                }
            }

            for(auto feat: fr->features)
            {
                if (feat->status == Feature::FeatureStatus::IMMATURE && feat->ip) {
                    // update the immature points
                    shared_ptr<ImmaturePoint> ph = feat->ip;
                    ph->traceOnforMF(fh, R, t, vvaff);

                    if (ph->lastTraceStatus == ImmaturePointStatus::IPS_GOOD) trace_good++;
                    if (ph->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION) trace_badcondition++;
                    if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OOB) trace_oob++;
                    if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER) trace_out++;
                    if (ph->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED) trace_skip++;
                    if (ph->lastTraceStatus == ImmaturePointStatus::IPS_UNINITIALIZED) trace_uninitialized++;
                    trace_total++;
                }
            }
        }
        printf("ADD: TRACE: %'d points. %'d (%.0f%%) good. %'d (%.0f%%) skip. %'d (%.0f%%) badcond. %'d (%.0f%%) oob. %'d (%.0f%%) out. %'d (%.0f%%) uninit.\n",
        trace_total,
        trace_good, 100*trace_good/(float)trace_total,
        trace_skip, 100*trace_skip/(float)trace_total,
        trace_badcondition, 100*trace_badcondition/(float)trace_total,
        trace_oob, 100*trace_oob/(float)trace_total,
        trace_out, 100*trace_out/(float)trace_total,
        trace_uninitialized, 100*trace_uninitialized/(float)trace_total);
    }

    void FullSystem::activatePointsMT() {
        // this will turn immature points into real points   根据当前窗口中已有的成熟点的数量ef->nPoints，设置激活阈值currentMinActDist
        if (ef->nPoints < setting_desiredPointDensity * 0.66)
            currentMinActDist -= 0.8;
        if (ef->nPoints < setting_desiredPointDensity * 0.8)
            currentMinActDist -= 0.5;
        else if (ef->nPoints < setting_desiredPointDensity * 0.9)
            currentMinActDist -= 0.2;
        else if (ef->nPoints < setting_desiredPointDensity)
            currentMinActDist -= 0.1;

        if (ef->nPoints > setting_desiredPointDensity * 1.5)
            currentMinActDist += 0.8;
        if (ef->nPoints > setting_desiredPointDensity * 1.3)
            currentMinActDist += 0.5;
        if (ef->nPoints > setting_desiredPointDensity * 1.15)
            currentMinActDist += 0.2;
        if (ef->nPoints > setting_desiredPointDensity)
            currentMinActDist += 0.1;

        if (currentMinActDist < 0) currentMinActDist = 0;
        if (currentMinActDist > 4) currentMinActDist = 4;

        auto newestFr = frames.back();
        vector<shared_ptr<FrameHessian>> frameHessians;
        for (auto fr: frames)
            frameHessians.push_back(fr->frameHessian);

        // make dist map  将所有的成熟点投影到当前帧，生成距离地图CoarseDistanceMap::makeDistanceMap 比如位置p 有一个投影点了，那么位置p 的值设为0，周围一圈像素设为１，再外面一圈设为２，以此类推，迭代进行
        coarseDistanceMap->makeK(Hcalib->mpCH);
        coarseDistanceMap->makeDistanceMap(frameHessians, newestFr->frameHessian);

        vector<shared_ptr<ImmaturePoint>> toOptimize;
        toOptimize.reserve(20000);

        // go through all active frames
        for (auto host: frameHessians) {
            if (host == newestFr->frameHessian)
                continue;

            SE3 fhToNew = newestFr->frameHessian->PRE_worldToCam * host->PRE_camToWorld;
            Mat33f KRKi = (coarseDistanceMap->K[1] * fhToNew.rotationMatrix().cast<float>() * coarseDistanceMap->Ki[0]);
            Vec3f Kt = (coarseDistanceMap->K[1] * fhToNew.translation().cast<float>());

            for (size_t i = 0; i < host->frame->features.size(); i++) {
                shared_ptr<Feature> &feat = host->frame->features[i];
                if (feat->status == Feature::FeatureStatus::IMMATURE && feat->ip) {

                    shared_ptr<Feature> &feat = host->frame->features[i];
                    shared_ptr<ImmaturePoint> &ph = host->frame->features[i]->ip;
                    ph->idxInImmaturePoints = i;

                    // delete points that have never been traced successfully, or that are outlier on the last trace.
                    if (!std::isfinite(ph->idepth_max) || ph->lastTraceStatus == IPS_OUTLIER) {
                        feat->status = Feature::FeatureStatus::OUTLIER;
                        feat->ReleaseImmature();
                        continue;
                    }

                    bool canActivate = (ph->lastTraceStatus == IPS_GOOD
                                        || ph->lastTraceStatus == IPS_SKIPPED
                                        || ph->lastTraceStatus == IPS_BADCONDITION
                                        || ph->lastTraceStatus == IPS_OOB)
                                       && ph->lastTracePixelInterval < 8
                                       && ph->quality > setting_minTraceQuality
                                       && (ph->idepth_max + ph->idepth_min) > 0;

                    if (!canActivate) {
                        // if point will be out afterwards, delete it instead.
                        if (ph->feature->host.lock()->frameHessian->flaggedForMarginalization ||
                            ph->lastTraceStatus == IPS_OOB) {
                            feat->status = Feature::FeatureStatus::OUTLIER;
                            feat->ReleaseImmature();
                        }
                        continue;
                    }

                    // see if we need to activate point due to distance map. 
                    Vec3f ptp = KRKi * Vec3f(feat->uv[0], feat->uv[1], 1) +
                                Kt * (0.5f * (ph->idepth_max + ph->idepth_min));    // 逆深度设为其最大值和最小值的平均，将其投影到当前帧
                    int u = ptp[0] / ptp[2] + 0.5f;
                    int v = ptp[1] / ptp[2] + 0.5f;

                    if ((u > 0 && v > 0 && u < wG[1] && v < hG[1])) {
                        // 计算其在距离地图上的值
                        float dist = coarseDistanceMap->fwdWarpedIDDistFinal[u + wG[1] * v] +
                                     (ptp[0] - floorf((float) (ptp[0])));

                        // NOTE: the shit my_type is used here   如果该值足够大 可以认为该点附近没有成熟点，所以将其加入待优化序列里
                        if (dist >= currentMinActDist * ph->my_type) {
                            coarseDistanceMap->addIntoDistFinal(u, v);
                            toOptimize.push_back(ph);
                        }
                    } else {
                        // drop it
                        feat->status = Feature::FeatureStatus::OUTLIER;
                        feat->ReleaseImmature();
                    }
                }
            }
        }

        vector<shared_ptr<PointHessian>> optimized;
        optimized.resize(toOptimize.size());

        // this will actually turn immature points into point hessians
        if (multiThreading) {
            threadReduce.reduce(
                bind(&FullSystem::activatePointsMT_Reductor, this, &optimized, &toOptimize, _1, _2, _3, _4), 0,
                toOptimize.size(), 50);
        } else {
            activatePointsMT_Reductor(&optimized, &toOptimize, 0, toOptimize.size(), 0, 0);     // 对待优化序列里的未成熟点进行优化FullSystem::activatePointsMT_Reductor，然后激活
        }

        for (size_t k = 0; k < toOptimize.size(); k++) {
            shared_ptr<PointHessian> newpoint = optimized[k];
            shared_ptr<ImmaturePoint> ph = toOptimize[k];

            if (newpoint != nullptr) {

                // remove the immature point
                ph->feature->status = Feature::FeatureStatus::VALID;

                ph->feature->point->mpPH = newpoint;
                ph->feature->ReleaseImmature();
                newpoint->takeData();

                for (auto r : newpoint->residuals)
                    ef->insertResidual(r);

            } else if (newpoint == nullptr || ph->lastTraceStatus == IPS_OOB) {

                ph->feature->status = Feature::FeatureStatus::OUTLIER;
                ph->feature->ReleaseImmature();

            }
        }
    }

    // for multi-fisheye
    void FullSystem::activatePointsMTforMF()
    {
        // this will turn immature points into real points   根据当前窗口中已有的成熟点的数量ef->nPoints，设置激活阈值currentMinActDist
        if (ef->nPoints < setting_desiredPointDensity * 0.66)
            currentMinActDist -= 0.8;
        if (ef->nPoints < setting_desiredPointDensity * 0.8)
            currentMinActDist -= 0.5;
        else if (ef->nPoints < setting_desiredPointDensity * 0.9)
            currentMinActDist -= 0.2;
        else if (ef->nPoints < setting_desiredPointDensity)
            currentMinActDist -= 0.1;

        if (ef->nPoints > setting_desiredPointDensity * 1.5)
            currentMinActDist += 0.8;
        if (ef->nPoints > setting_desiredPointDensity * 1.3)
            currentMinActDist += 0.5;
        if (ef->nPoints > setting_desiredPointDensity * 1.15)
            currentMinActDist += 0.2;
        if (ef->nPoints > setting_desiredPointDensity)
            currentMinActDist += 0.1;

        if (currentMinActDist < 0) currentMinActDist = 0;
        if (currentMinActDist > 4) currentMinActDist = 4;

        auto newestFr = frames.back();
        vector<shared_ptr<FrameHessian>> frameHessians;
        for (auto fr: frames)
            frameHessians.push_back(fr->frameHessian);

        // make dist map  将所有的成熟点投影到当前帧，生成距离地图CoarseDistanceMap::makeDistanceMap 比如位置p 有一个投影点了，那么位置p 的值设为0，周围一圈像素设为１，再外面一圈设为２，以此类推，迭代进行
        //coarseDistanceMap->makeK(Hcalib->mpCH);
        coarseDistanceMap->makeDistanceMapforMF(frameHessians, newestFr->frameHessian);

        vector<shared_ptr<ImmaturePoint>> toOptimize;
        toOptimize.reserve(20000);

        // go through all active frames 处理未成熟点, 激活/删除/跳过 遍历滑动窗口内所有帧  遍历所有的未成熟点, 满足条件 将其投影到当前帧，然后考虑其在距离地图上的值，如果该值足够大，则加入优化，否则则删除
        for (auto host: frameHessians) 
        {
            if (host == newestFr->frameHessian)
                continue;

            SE3 fhToNew = newestFr->frameHessian->PRE_worldToCam * host->PRE_camToWorld;
            Mat33f R = fhToNew.rotationMatrix().cast<float>();
            Vec3f t = fhToNew.translation().cast<float>();
            // 遍历每帧上面的 所有未成熟点
            for (size_t i = 0; i < host->frame->features.size(); i++) 
            {
                shared_ptr<Feature> &feat = host->frame->features[i];
                if (feat->status == Feature::FeatureStatus::IMMATURE && feat->ip) 
                {

                    shared_ptr<Feature> &feat = host->frame->features[i];
                    shared_ptr<ImmaturePoint> &ph = host->frame->features[i]->ip;
                    ph->idxInImmaturePoints = i;

                    // delete points that have never been traced successfully, or that are outlier on the last trace.
                    if (!std::isfinite(ph->idepth_max) || ph->lastTraceStatus == IPS_OUTLIER) 
                    {
                        feat->status = Feature::FeatureStatus::OUTLIER;
                        feat->ReleaseImmature();
                        continue;
                    }

                    bool canActivate = (ph->lastTraceStatus == IPS_GOOD
                                        || ph->lastTraceStatus == IPS_SKIPPED
                                        || ph->lastTraceStatus == IPS_BADCONDITION
                                        || ph->lastTraceStatus == IPS_OOB)
                                       && ph->lastTracePixelInterval < 8
                                       && ph->quality > setting_minTraceQuality
                                       && (ph->idepth_max + ph->idepth_min) > 0;

                    if (!canActivate) 
                    {
                        // if point will be out afterwards, delete it instead.
                        if (ph->feature->host.lock()->frameHessian->flaggedForMarginalization ||
                            ph->lastTraceStatus == IPS_OOB) 
                        {
                            feat->status = Feature::FeatureStatus::OUTLIER;
                            feat->ReleaseImmature();
                        }
                        continue;
                    }

                    // see if we need to activate point due to distance map. 
                    // Vec3f ptp = KRKi * Vec3f(feat->uv[0], feat->uv[1], 1) +
                    //             Kt * (0.5f * (ph->idepth_max + ph->idepth_min));    // 逆深度设为其最大值和最小值的平均，将其投影到当前帧
                    float SphereRadius = UMF->GetSphereRadius();

                    Vec3f pt = R * Vec3f(feat->xyzs[0]/SphereRadius, feat->xyzs[1]/SphereRadius, feat->xyzs[2]/SphereRadius) / (0.5f * ph->idepth_max + ph->idepth_min) + t ;   
                    
                    float S_norm_pt = SphereRadius/pt.norm();

                    float xs = S_norm_pt*pt(0);
                    float ys = S_norm_pt*pt(1);
                    float zs = S_norm_pt*pt(2);

                    double Ku, Kv;
                    int tocamnum;

                    UMF->LadybugReprojectSpherePtToRectify(xs, ys, zs, &tocamnum, &Ku, &Kv, 1);
                    int u = Ku + 0.5f;
                    int v = Kv + 0.5f;

                    if ((u > 0 && v > 0 && u < UMF->wPR[1] && v < UMF->hPR[1]) && newestFr->frameHessian->vmask[1][tocamnum][u + v*UMF->wPR[1]] != 0) 
                    {
                        // 计算其在距离地图上的值
                        float dist = coarseDistanceMap->vfwdWarpedIDDistFinal[tocamnum][u + UMF->wPR[1] * v] +
                                     (Ku - floorf((float) Ku));

                        // NOTE: the shit my_type is used here   如果该值足够大 可以认为该点附近没有成熟点，所以将其加入待优化序列里
                        if (dist >= currentMinActDist * ph->my_type) {
                            //coarseDistanceMap->addIntoDistFinal(u, v);
                            coarseDistanceMap->addIntoDistFinalforMF(u, v, tocamnum);
                            toOptimize.push_back(ph);
                        }
                    } else {
                        // drop it
                        feat->status = Feature::FeatureStatus::OUTLIER;
                        feat->ReleaseImmature();
                    }
                }
            }
        }

        vector<shared_ptr<PointHessian>> optimized;
        optimized.resize(toOptimize.size());

        // this will actually turn immature points into point hessians
        if (multiThreading) 
        {
            threadReduce.reduce(
                bind(&FullSystem::activatePointsMT_ReductorforMF, this, &optimized, &toOptimize, _1, _2, _3, _4), 0,
                toOptimize.size(), 50);
        } 
        else 
        {
            activatePointsMT_ReductorforMF(&optimized, &toOptimize, 0, toOptimize.size(), 0, 0);     // 对待优化序列里的未成熟点进行优化FullSystem::activatePointsMT_Reductor，然后激活 优化上一步挑出来的未成熟点, 进行逆深度优化, 并得到pointhessian
        }
        // 把PointHessian加入到能量函数, 删除收敛的未成熟点, 或不好的点
        for (size_t k = 0; k < toOptimize.size(); k++) 
        {
            shared_ptr<PointHessian> newpoint = optimized[k];
            shared_ptr<ImmaturePoint> ph = toOptimize[k];

            if (newpoint != nullptr) 
            {

                // remove the immature point
                ph->feature->status = Feature::FeatureStatus::VALID;

                ph->feature->point->mpPH = newpoint;
                ph->feature->ReleaseImmature();
                newpoint->takeData();

                for (auto r : newpoint->residuals)
                    ef->insertResidual(r);

            } 
            else if (newpoint == nullptr || ph->lastTraceStatus == IPS_OOB) 
            {

                ph->feature->status = Feature::FeatureStatus::OUTLIER;
                ph->feature->ReleaseImmature();

            }
        }
    }

    void FullSystem::activatePointsMT_Reductor(
        std::vector<shared_ptr<PointHessian>> *optimized,
        std::vector<shared_ptr<ImmaturePoint>> *toOptimize,
        int min, int max, Vec10 *stats, int tid) {

        vector<shared_ptr<ImmaturePointTemporaryResidual>> tr(frames.size(), nullptr);

        for (auto &t:tr) {
            // create residual
            t = shared_ptr<ImmaturePointTemporaryResidual>(new ImmaturePointTemporaryResidual());
        }

        for (int k = min; k < max; k++) {
            (*optimized)[k] = optimizeImmaturePoint((*toOptimize)[k], 1, tr);
        }
    }

    // for multi-fisheye
    void FullSystem::activatePointsMT_ReductorforMF(
        std::vector<shared_ptr<PointHessian>> *optimized,
        std::vector<shared_ptr<ImmaturePoint>> *toOptimize,
        int min, int max, Vec10 *stats, int tid) 
    {

        vector<shared_ptr<ImmaturePointTemporaryResidual>> tr(frames.size(), nullptr);

        for (auto &t:tr) 
        {
            // create residual  窗口内所有帧
            t = shared_ptr<ImmaturePointTemporaryResidual>(new ImmaturePointTemporaryResidual());
        }

        for (int k = min; k < max; k++) 
        {
            (*optimized)[k] = optimizeImmaturePointforMF((*toOptimize)[k], 1, tr);
        }
    }


    void FullSystem::flagPointsForRemoval() {
        assert(EFIndicesValid);
        std::vector<shared_ptr<FrameHessian>>
            fhsToMargPoints;

        for (int i = 0; i < (int) frames.size(); i++)
            if (frames[i]->frameHessian->flaggedForMarginalization)
                fhsToMargPoints.push_back(frames[i]->frameHessian);

        int flag_oob = 0, flag_in = 0, flag_inin = 0, flag_nores = 0;

        // go through all active frames
        for (auto &fr : frames) {
            shared_ptr<FrameHessian> host = fr->frameHessian;
            for (auto &feat: fr->features) {
                if (feat->status == Feature::FeatureStatus::VALID &&
                    feat->point->status == Point::PointStatus::ACTIVE) {

                    shared_ptr<PointHessian> ph = feat->point->mpPH;

                    if (ph->idepth_scaled < 0 || ph->residuals.size() == 0) {
                        // no residuals or idepth invalid
                        ph->point->status = Point::PointStatus::OUTLIER;
                        feat->status = Feature::FeatureStatus::OUTLIER;
                        flag_nores++;

                    } else if (ph->isOOB(fhsToMargPoints) || host->flaggedForMarginalization) { // 一个是该点构成的误差项较少,该点的Host帧被标记为边缘化 
                        // the point goes out the boundary, or the host frame is marged
                        flag_oob++;
                        if (ph->isInlierNew()) {
                            flag_in++;
                            int ngoodRes = 0;
                            for (auto r: ph->residuals) {
                                r->resetOOB();
                                r->linearize(this->Hcalib->mpCH);
                                r->isLinearized = false;
                                r->applyRes(true);
                                if (r->isActive()) {
                                    r->fixLinearizationF(this->ef);
                                    ngoodRes++;
                                }
                            }

                            if (ph->idepth_hessian > setting_minIdepthH_marg) {
                                // idepth is good, margin it.
                                flag_inin++;
                                ph->point->status = Point::PointStatus::MARGINALIZED;
                            } else {
                                // idepth not good, drop it.
                                ph->point->status = Point::PointStatus::OUT;
                            }
                        } else {
                            // not active, drop it.
                            ph->point->status = Point::PointStatus::OUT;
                        }
                    }
                }
            }
        }

        LOG(INFO) << "Flag: nores: " << flag_nores << ", oob: " << flag_oob << ", marged: " << flag_inin << endl;
    }

    // for multi-fisheye 标记要移除点的状态, 边缘化or丢掉
    void FullSystem::flagPointsForRemovalforMF() 
    {
        assert(EFIndicesValid);
        std::vector<shared_ptr<FrameHessian>> fhsToMargPoints;

        for (int i = 0; i < (int) frames.size(); i++)
            if (frames[i]->frameHessian->flaggedForMarginalization)
                fhsToMargPoints.push_back(frames[i]->frameHessian);

        int flag_oob = 0, flag_in = 0, flag_inin = 0, flag_nores = 0;

        // go through all active frames
        for (auto &fr : frames) 
        {
            shared_ptr<FrameHessian> host = fr->frameHessian;
            for (auto &feat: fr->features) 
            {
                if (feat->status == Feature::FeatureStatus::VALID &&
                    feat->point->status == Point::PointStatus::ACTIVE) 
                {
                    shared_ptr<PointHessian> ph = feat->point->mpPH;

                    // 丢掉相机后面, 没有残差的点
                    if (ph->idepth_scaled < 0 || ph->residuals.size() == 0) 
                    {
                        // no residuals or idepth invalid
                        ph->point->status = Point::PointStatus::OUTLIER;
                        feat->status = Feature::FeatureStatus::OUTLIER;
                        flag_nores++;

                    } // 把边缘化的帧上的点, 以及受影响较大的点标记为边缘化or删除
                    else if (ph->isOOB(fhsToMargPoints) || host->flaggedForMarginalization) 
                    { // 1该点构成的误差项较少 isOOB,点投影下来没有落在视场范围内,没有构成残差,2该点的Host帧被标记为边缘化 
                        // the point goes out the boundary, or the host frame is marged
                        flag_oob++;
                        //* 如果是一个内点, 则把残差在当前状态线性化, 并计算到零点残差
                        if (ph->isInlierNew()) 
                        {
                            flag_in++;
                            int ngoodRes = 0;
                            for (auto r: ph->residuals) 
                            {
                                // 重新计算该点的误差，若小于阈值则设为active
                                r->resetOOB();
                                r->linearizeforMF(this->UMF);
                                r->isLinearized = false;
                                r->applyRes(true);
                                // 如果是激活(可参与优化)的残差, 则给fix住, 计算res_toZeroF
                                if (r->isActive()) 
                                {   
                                    r->fixLinearizationFforMF(this->ef);
                                    ngoodRes++;
                                }
                            }

                            //* 如果逆深度的协方差很大直接扔掉, 小的边缘化掉
                            if (ph->idepth_hessian > setting_minIdepthH_marg) 
                            {
                                // idepth is good, margin it.
                                flag_inin++;
                                ph->point->status = Point::PointStatus::MARGINALIZED;
                            } 
                            else 
                            {
                                // idepth not good, drop it.
                                ph->point->status = Point::PointStatus::OUT;
                            }
                        } 
                        //* 不是内点直接扔掉
                        else 
                        {
                            // not active, drop it.
                            ph->point->status = Point::PointStatus::OUT;
                        }
                    }
                }
            }
        }

        LOG(INFO) << "Flag: nores: " << flag_nores << ", oob: " << flag_oob << ", marged: " << flag_inin << endl;
    }

    void FullSystem::makeNewTraces(shared_ptr<FrameHessian> newFrame, float *gtDepth) {

        if (setting_pointSelection == 1) {
            LOG(INFO) << "using LDSO point selection strategy " << endl;
            newFrame->frame->features.reserve(setting_desiredImmatureDensity);
            detector.DetectCorners(setting_desiredImmatureDensity, newFrame->frame);
            for (auto &feat: newFrame->frame->features) {
                // create a immature point
                feat->ip = shared_ptr<ImmaturePoint>(
                    new ImmaturePoint(newFrame->frame, feat, 1, Hcalib->mpCH));
            }
            LOG(INFO) << "new features features created: " << newFrame->frame->features.size() << endl;
        } else if (setting_pointSelection == 0) {
            LOG(INFO) << "using original DSO point selection strategy" << endl;
            pixelSelector->allowFast = true;
            int numPointsTotal = pixelSelector->makeMaps(newFrame, selectionMap, setting_desiredImmatureDensity);
            newFrame->frame->features.reserve(numPointsTotal);

            for (int y = patternPadding + 1; y < hG[0] - patternPadding - 2; y++)
                for (int x = patternPadding + 1; x < wG[0] - patternPadding - 2; x++) {
                    int i = x + y * wG[0];
                    if (selectionMap[i] == 0) continue;

                    shared_ptr<Feature> feat(new Feature(x, y, newFrame->frame));
                    feat->ip = shared_ptr<ImmaturePoint>(
                        new ImmaturePoint(newFrame->frame, feat, selectionMap[i], Hcalib->mpCH));
                    if (!std::isfinite(feat->ip->energyTH)) {
                        feat->ReleaseAll();
                        continue;
                    } else
                        newFrame->frame->features.push_back(feat);
                }
            LOG(INFO) << "new features features created: " << newFrame->frame->features.size() << endl;
        } else if (setting_pointSelection == 2) {
            // random pick
            LOG(INFO) << "using random point selection strategy" << endl;
            cv::RNG rng;
            newFrame->frame->features.reserve(setting_desiredImmatureDensity);
            for (int i = 0; i < setting_desiredImmatureDensity; i++) {
                int x = rng.uniform(20, wG[0] - 20);
                int y = rng.uniform(20, hG[0] - 20);
                shared_ptr<Feature> feat(new Feature(x, y, newFrame->frame));
                feat->ip = shared_ptr<ImmaturePoint>(
                    new ImmaturePoint(newFrame->frame, feat, 1, Hcalib->mpCH));
                if (!std::isfinite(feat->ip->energyTH)) {
                    feat->ReleaseAll();
                    continue;
                } else
                    newFrame->frame->features.push_back(feat);
            }
            LOG(INFO) << "new features features created: " << newFrame->frame->features.size() << endl;
        }
    }

    // for multi-fisheye
    void FullSystem::makeNewTracesforMF(shared_ptr<FrameHessian> newFrame, float *gtDepth) 
    {
        int camnums = UMF->camNums;
        if (setting_pointSelection == 1) 
        {
            LOG(INFO) << "using LDSO point selection strategy " << endl;
            newFrame->frame->features.reserve(setting_desiredImmatureDensity);
            for(int n = 0; n < camnums; n++)
            {
                detector.DetectCornersforMF(setting_desiredImmatureDensity, newFrame->frame, n,  UMF);
                for (auto &feat: newFrame->frame->features) 
                {
                    // create a immature point
                    feat->ip = shared_ptr<ImmaturePoint>(
                        new ImmaturePoint(newFrame->frame, feat, 1, n, UMF));
                }
            }
            LOG(INFO) << "new features features created: " << newFrame->frame->features.size() << endl;
        } 
        else if (setting_pointSelection == 0) 
        {
            LOG(INFO) << "using original DSO point selection strategy" << endl;
            pixelSelector->allowFast = true;

            int numPointsTotal = 0;
            for(int n = 0; n < camnums; n++)
            {
                numPointsTotal += pixelSelector->makeMapsforMF(newFrame, n, selectionMap, setting_desiredImmatureDensity);
            
                newFrame->frame->features.reserve(numPointsTotal);

                for (int y = patternPadding + 1; y < UMF->hPR[0] - patternPadding - 2; y++)
                    for (int x = patternPadding + 1; x < UMF->wPR[0] - patternPadding - 2; x++) 
                    {
                        int i = x + y * UMF->wPR[0];
                        if (selectionMap[i] == 0) continue;

                        double X,Y,Z;
                        UMF->LadybugProjectRectifyPtToSphere(n, x, y, &X, &Y, &Z, 0);
                        shared_ptr<Feature> feat(new Feature(x, y, X, Y, Z, newFrame->frame, n));
                        //shared_ptr<Feature> feat(new Feature(x, y, newFrame->frame));
                        feat->ip = shared_ptr<ImmaturePoint>(
                            new ImmaturePoint(newFrame->frame, feat, selectionMap[i], n, UMF));
                        if (!std::isfinite(feat->ip->energyTH)) {
                            feat->ReleaseAll();
                            continue;
                        } else
                            newFrame->frame->features.push_back(feat);
                    }
            }
            LOG(INFO) << "new features features created: " << newFrame->frame->features.size() << endl;
        } else if (setting_pointSelection == 2) 
        {
            // random pick
            LOG(INFO) << "using random point selection strategy" << endl;
            cv::RNG rng;
            newFrame->frame->features.reserve(setting_desiredImmatureDensity * camnums);
            for(int n = 0; n < UMF->camNums; n++)
            {
                for (int i = 0; i < setting_desiredImmatureDensity; i++) 
                {
                    int x = rng.uniform(20, UMF->wPR[0] - 20);
                    int y = rng.uniform(20, UMF->hPR[0] - 20);
                    double X,Y,Z;
                    UMF->LadybugProjectRectifyPtToSphere(n, x, y, &X, &Y, &Z, 0);
                    shared_ptr<Feature> feat(new Feature(x, y, X, Y, Z, newFrame->frame, n));
                    feat->ip = shared_ptr<ImmaturePoint>(
                        new ImmaturePoint(newFrame->frame, feat, 1, n, UMF));
                    if (!std::isfinite(feat->ip->energyTH)) {
                        feat->ReleaseAll();
                        continue;
                    } else
                        newFrame->frame->features.push_back(feat);
                }
            }
            LOG(INFO) << "new features features created: " << newFrame->frame->features.size() << endl;
        }
    }

    void FullSystem::initializeFromInitializer(shared_ptr<FrameHessian> newFrame) {
        unique_lock<mutex> lock(mapMutex);

        shared_ptr<FrameHessian> firstFrame = coarseInitializer->firstFrame;
        shared_ptr<Frame> fr = firstFrame->frame;
        firstFrame->idx = frames.size();   // slide window id

        frames.push_back(fr);
        firstFrame->frameID = globalMap->NumFrames();
        ef->insertFrame(firstFrame, Hcalib->mpCH);
        setPrecalcValues();

        fr->features.reserve(wG[0] * hG[0] * 0.2f);

        float sumID = 1e-5, numID = 1e-5;
        for (int i = 0; i < coarseInitializer->numPoints[0]; i++) {
            sumID += coarseInitializer->points[0][i].iR;
            numID++;
        }
        float rescaleFactor = 1 / (sumID / numID);   // 平均深度

        // randomly sub-select the points I need.
        float keepPercentage = setting_desiredPointDensity / coarseInitializer->numPoints[0];

        LOG(INFO) << "Initialization: keep " << 100 * keepPercentage << " (need " << setting_desiredPointDensity
                  << ", have " << coarseInitializer->numPoints[0] << ")!" << endl;

        // Create features in the first frame.
        for (size_t i = 0; i < size_t(coarseInitializer->numPoints[0]); i++) {

            if (rand() / (float) RAND_MAX > keepPercentage)
                continue;
            Pnt *point = coarseInitializer->points[0] + i;

            shared_ptr<Feature> feat(new Feature(point->u + 0.5f, point->v + 0.5f, firstFrame->frame));
            feat->ip = shared_ptr<ImmaturePoint>(
                new ImmaturePoint(firstFrame->frame, feat, point->my_type, Hcalib->mpCH));

            if (!std::isfinite(feat->ip->energyTH)) {
                feat->ReleaseImmature();
                continue;
            }

            feat->CreateFromImmature();
            shared_ptr<PointHessian> ph = feat->point->mpPH;
            if (!std::isfinite(ph->energyTH)) {
                feat->ReleaseMapPoint();
                continue;
            }
            feat->ReleaseImmature();    // no longer needs the immature part
            fr->features.push_back(feat);

            ph->setIdepthScaled(point->iR * rescaleFactor);    // 保证所有点深度均值为1
            ph->setIdepthZero(ph->idepth);
            ph->hasDepthPrior = true;
            ph->point->status = Point::PointStatus::ACTIVE;
            ph->takeData(); // set the idepth into optimization

        }

        SE3 firstToNew = coarseInitializer->thisToNext;
        firstToNew.translation() /= rescaleFactor;   // 设置尺度

        // really no lock required, as we are initializing.
        {
            unique_lock<mutex> crlock(shellPoseMutex);
            firstFrame->setEvalPT_scaled(fr->getPose(), firstFrame->frame->aff_g2l);
            newFrame->frame->setPose(firstToNew);
            newFrame->setEvalPT_scaled(newFrame->frame->getPose(), newFrame->frame->aff_g2l);
        }

        initialized = true;
        globalMap->AddKeyFrame(fr);
        LOG(INFO) << "Initialized from initializer, points: " << firstFrame->frame->features.size() << endl;
    }

    // for multi-fisheye
    void FullSystem::initializeFromInitializerforMF(shared_ptr<FrameHessian> newFrame)
    {
         unique_lock<mutex> lock(mapMutex);

        shared_ptr<FrameHessian> firstFrame = coarseInitializer->firstFrame;
        shared_ptr<Frame> fr = firstFrame->frame;
        firstFrame->idx = frames.size();   // slide window id

        frames.push_back(fr);
        firstFrame->frameID = globalMap->NumFrames();
		ef->insertFrameforMF(firstFrame);
	    setPrecalcValuesforMF();   

		fr->features.reserve(UMF->camNums * UMF->wPR[0] * UMF->hPR[0] * 0.2f);

		float sumID = 1e-5, numID = 1e-5;
		for(int n = 0; n <UMF->camNums; n++)
		{
			for (int i = 0; i < coarseInitializer->vnumPoints[0][n]; i++) 
			{
	            		sumID += coarseInitializer->vpoints[0][n][i].iR;
	            		numID++;
	        	}
		}
	        
	    float rescaleFactor = 1 / (sumID / numID);   // 平均深度

		LOG(INFO) << "Initialization: keep " <<endl;

		for(int n = 0; n <UMF->camNums; n++)
		{
            float keepPercentage = setting_desiredPointDensity / coarseInitializer->vnumPoints[0][n];
            LOG(INFO) << "camera" << n  << ": " << 100 * keepPercentage << " (need " << setting_desiredPointDensity
	                  << ", have " << coarseInitializer->vnumPoints[0][n] << ")!" << endl;
			for (size_t i = 0; i < size_t(coarseInitializer->vnumPoints[0][n]); i++)
				{
					if (rand() / (float) RAND_MAX > keepPercentage)
	                continue;
	            	Pnt *point = coarseInitializer->vpoints[0][n] + i;
                    double X,Y,Z;
                    UMF->LadybugProjectRectifyPtToSphere(n, point->u + 0.5f, point->v + 0.5f, &X, &Y, &Z, 0);
	            	shared_ptr<Feature> feat(new Feature(point->u + 0.5f, point->v + 0.5f, X, Y, Z, firstFrame->frame, n));   // 初始化点 host为第一帧
		            feat->ip = shared_ptr<ImmaturePoint>(
		                new ImmaturePoint(firstFrame->frame, feat, point->my_type, n, UMF));   

		            if (!std::isfinite(feat->ip->energyTH)) {
		                feat->ReleaseImmature();
		                continue;
		            }

		            feat->CreateFromImmature();
		            shared_ptr<PointHessian> ph = feat->point->mpPH;
		            if (!std::isfinite(ph->energyTH)) {
		                feat->ReleaseMapPoint();
		                continue;
		            }
		            feat->ReleaseImmature();    // no longer needs the immature part
		            fr->features.push_back(feat);

		            ph->setIdepthScaled(point->iR * rescaleFactor);    // 保证所有点深度均值为1
		            ph->setIdepthZero(ph->idepth);
		            ph->hasDepthPrior = true;
		            ph->point->status = Point::PointStatus::ACTIVE;
		            ph->takeData(); // set the idepth into optimization
				}
			
		}

		SE3 firstToNew = coarseInitializer->thisToNext;
        firstToNew.translation() /= rescaleFactor;   // 设置尺度

        // really no lock required, as we are initializing.
        {
            unique_lock<mutex> crlock(shellPoseMutex);
            firstFrame->setEvalPT_scaledforMF(fr->getPose(), firstFrame->frame->vaff_g2l);   // nullspaces_affine 未修改
            newFrame->frame->setPose(firstToNew);
            newFrame->setEvalPT_scaledforMF(newFrame->frame->getPose(), newFrame->frame->vaff_g2l);
        }

        initialized = true;
        globalMap->AddKeyFrame(fr);
        LOG(INFO) << "Initialized from initializer, points: " << firstFrame->frame->features.size() << endl;
    }

    void FullSystem::removeOutliers() {
        int numPointsDropped = 0;
        for (auto &fr: frames) {
            for (auto &feat: fr->features) {
                if (feat->status == Feature::FeatureStatus::VALID && feat->point
                    && feat->point->status == Point::PointStatus::ACTIVE) {

                    shared_ptr<PointHessian> ph = feat->point->mpPH;
                    if (ph->residuals.empty()) {
                        ph->point->status = Point::PointStatus::OUTLIER;
                        feat->status = Feature::FeatureStatus::OUTLIER;
                        numPointsDropped++;
                    }
                }
            }
        }

        LOG(INFO) << "remove outliers done, outliers: " << numPointsDropped << endl;
        ef->dropPointsF();
    }

    void FullSystem::removeOutliersforMF() 
    {
        int numPointsDropped = 0;
        for (auto &fr: frames) {
            for (auto &feat: fr->features) 
            {
                if (feat->status == Feature::FeatureStatus::VALID && feat->point
                    && feat->point->status == Point::PointStatus::ACTIVE) 
                {
                    shared_ptr<PointHessian> ph = feat->point->mpPH;
                    if (ph->residuals.empty())    // 若该点残差数为0 则丢弃
                    {
                        ph->point->status = Point::PointStatus::OUTLIER;
                        feat->status = Feature::FeatureStatus::OUTLIER;
                        numPointsDropped++;
                    }
                }
            }
        }

        LOG(INFO) << "remove outliers done, outliers: " << numPointsDropped << endl;
        ef->dropPointsF();
    }

    void FullSystem::setPrecalcValues() {
        for (auto &fr: frames) {
            fr->frameHessian->targetPrecalc.resize(frames.size());
            for (size_t i = 0; i < frames.size(); i++)
                fr->frameHessian->targetPrecalc[i].Set(fr->frameHessian, frames[i]->frameHessian, Hcalib->mpCH);
        }

        ef->setDeltaF(Hcalib->mpCH);
    }

    // for multi-Fisheye
    void FullSystem::setPrecalcValuesforMF()
    {
        for (auto &fr: frames) 
        {
            fr->frameHessian->targetPrecalc.resize(frames.size());
            for (size_t i = 0; i < frames.size(); i++)
                fr->frameHessian->targetPrecalc[i].SetforMF(fr->frameHessian, frames[i]->frameHessian, UMF);  //该函数设置关键帧间相对（ host 到 target）的状态变量
        }

        ef->setDeltaFforMF();   
    }

    void FullSystem::solveSystem(int iteration, double lambda) {
        ef->lastNullspaces_forLogging = getNullspaces(
            ef->lastNullspaces_pose,
            ef->lastNullspaces_scale,
            ef->lastNullspaces_affA,
            ef->lastNullspaces_affB);
        ef->solveSystemF(iteration, lambda, Hcalib->mpCH);
    }

    void FullSystem::solveSystemforMF(int iteration, double lambda)
    {
        //得到各个状态的零空间
        ef->lastNullspaces_forLogging = getNullspacesforMF(
            ef->lastNullspaces_pose,
            ef->lastNullspaces_scale);
        
        ef->solveSystemFforMF(iteration, lambda);
    }

    Vec3 FullSystem::linearizeAll(bool fixLinearization) {

        double lastEnergyP = 0;
        double lastEnergyR = 0;
        double num = 0;

        std::vector<shared_ptr<PointFrameResidual>>
            toRemove[NUM_THREADS];
        for (int i = 0; i < NUM_THREADS; i++)
            toRemove[i].clear();

        if (multiThreading) {
            threadReduce.reduce(
                bind(&FullSystem::linearizeAll_Reductor, this, fixLinearization, toRemove, _1, _2, _3, _4),
                0, activeResiduals.size(), 0);
            lastEnergyP = threadReduce.stats[0];
        } else {
            Vec10 stats;
            linearizeAll_Reductor(fixLinearization, toRemove, 0, activeResiduals.size(), &stats, 0);
            lastEnergyP = stats[0];
        }

        setNewFrameEnergyTH();

        if (fixLinearization) {

            for (auto r : activeResiduals) {
                shared_ptr<PointHessian> ph = r->point.lock();
                if (ph->lastResiduals[0].first == r)
                    ph->lastResiduals[0].second = r->state_state;
                else if (ph->lastResiduals[1].first == r)
                    ph->lastResiduals[1].second = r->state_state;
            }

            int nResRemoved = 0;
            for (int i = 0; i < NUM_THREADS; i++) {
                for (auto r : toRemove[i]) {
                    shared_ptr<PointHessian> ph = r->point.lock();

                    if (ph->lastResiduals[0].first == r)
                        ph->lastResiduals[0].first = 0;
                    else if (ph->lastResiduals[1].first == r)
                        ph->lastResiduals[1].first = 0;

                    ef->dropResidual(r);    // this will actually remove the residual
                    nResRemoved++;
                }
            }
        }
        return Vec3(lastEnergyP, lastEnergyR, num);
    }

    // for multi-fisheye 计算相关的导数 对残差进行线性化, 并去掉不在图像内, 并且残差大的
    Vec3 FullSystem::linearizeAllforMF(bool fixLinearization)
    {
         double lastEnergyP = 0;
        double lastEnergyR = 0;
        double num = 0;

        std::vector<shared_ptr<PointFrameResidual>>
            toRemove[NUM_THREADS];
        for (int i = 0; i < NUM_THREADS; i++)
            toRemove[i].clear();

        // if (multiThreading) 
        if (false)
        {
            threadReduce.reduce(
                bind(&FullSystem::linearizeAll_ReductorforMF, this, fixLinearization, toRemove, _1, _2, _3, _4),
                0, activeResiduals.size(), 0);
            lastEnergyP = threadReduce.stats[0];
        } 
        else 
        {
            Vec10 stats;
            linearizeAll_ReductorforMF(fixLinearization, toRemove, 0, activeResiduals.size(), &stats, 0);
            lastEnergyP = stats[0];
        }
        // 计算当前最新帧的能量阈值
        setNewFrameEnergyTH();

        if (fixLinearization) 
        {
            // 前面线性化, apply之后更新了state_state, 如果有相同的, 就更新状态
            for (auto r : activeResiduals) 
            {
                shared_ptr<PointHessian> ph = r->point.lock();
                if (ph->lastResiduals[0].first == r)
                    ph->lastResiduals[0].second = r->state_state;
                else if (ph->lastResiduals[1].first == r)
                    ph->lastResiduals[1].second = r->state_state;
            }
            // residual创建时候都创建, 再去掉不好的
            int nResRemoved = 0;
            for (int i = 0; i < NUM_THREADS; i++) 
            {
                for (auto r : toRemove[i]) 
                {
                    shared_ptr<PointHessian> ph = r->point.lock();

                    if (ph->lastResiduals[0].first == r)
                        ph->lastResiduals[0].first = 0;
                    else if (ph->lastResiduals[1].first == r)
                        ph->lastResiduals[1].first = 0;

                    ef->dropResidual(r);    // this will actually remove the residual
                    nResRemoved++;
                }
            }
        }
        return Vec3(lastEnergyP, lastEnergyR, num);
    }

    void FullSystem::linearizeAll_Reductor(
        bool fixLinearization, std::vector<shared_ptr<PointFrameResidual>>

    *toRemove,
        int min,
        int max, Vec10
        *stats,
        int tid
    ) {

        for (
            int k = min;
            k < max;
            k++) {
            shared_ptr<PointFrameResidual> r = activeResiduals[k];
            (*stats)[0] += r->
                linearize(Hcalib
                              ->mpCH);

            if (fixLinearization) {
                r->applyRes(true);

                if (r->

                    isActive()

                    ) {
                    if (r->isNew) {
                        shared_ptr<PointHessian> p = r->point.lock();
                        shared_ptr<FrameHessian> host = r->host.lock();
                        shared_ptr<FrameHessian> target = r->target.lock();
                        Vec3f ptp_inf = host->targetPrecalc[target->idx].PRE_KRKiTll *
                                        Vec3f(p->u, p->v, 1);    // projected point assuming infinite depth.
                        Vec3f ptp = ptp_inf + host->targetPrecalc[target->idx].PRE_KtTll *
                                              p->idepth_scaled;    // projected point with real depth.
                        float relBS = 0.01 * ((ptp_inf.head<2>() / ptp_inf[2]) -
                                              (ptp.head<2>() / ptp[2])).norm();    // 0.01 = one pixel.
                        if (relBS > p->maxRelBaseline)
                            p->
                                maxRelBaseline = relBS;

                        p->numGoodResiduals++;
                    }
                } else {
                    toRemove[tid].
                        push_back(activeResiduals[k]);
                }
            }
        }
    }

    // for multi-fisheye [true是applyRes, 并去掉不好的残差] [false不进行固定线性化]
    void FullSystem::linearizeAll_ReductorforMF(bool fixLinearization, std::vector<shared_ptr<PointFrameResidual>> *toRemove, 
                    int min, int max, Vec10 *stats, int tid)
    {
        for (int k = min; k < max; k++) 
        {
            shared_ptr<PointFrameResidual> r = activeResiduals[k];
            (*stats)[0] += r->linearizeforMF(UMF); // 线性化得到能量  求目标函数对各个变量的导数

            if (fixLinearization)   // 固定线性化（优化后执行）
            {
                r->applyRes(true);  //  更新残差的状态(好坏)

                if (r->isActive()) // 残差是in的
                {
                    if (r->isNew) 
                    {
                        // 无穷远点
                        shared_ptr<PointHessian> p = r->point.lock();
                        shared_ptr<FrameHessian> host = r->host.lock();
                        shared_ptr<FrameHessian> target = r->target.lock();
                        float xs = p->xs;
                        float ys = p->ys;
                        float zs = p->zs;
                        float SphereRadius = UMF->GetSphereRadius();

                        Vec3f ptp_inf = host->targetPrecalc[target->idx].PRE_RTll *
                                        Vec3f(xs/SphereRadius, ys/SphereRadius, zs/SphereRadius)/0.001 + host->targetPrecalc[target->idx].PRE_tTll;    // projected point assuming infinite depth.
                        float S_norm_pt = SphereRadius/ptp_inf.norm();

                        float xs1 = S_norm_pt*ptp_inf(0);
                        float ys1 = S_norm_pt*ptp_inf(1);
                        float zs1 = S_norm_pt*ptp_inf(2);

                        // double Ku1, Kv1;
                        // int tocamnum;

                        // //UMF->LadybugReprojectSpherePtToFishEyeImg(xs, ys, zs, &tocamnum, &Ku, &Kv, lvl);
                        // UMF->LadybugReprojectSpherePtToRectify(xs1, ys1, zs1, &tocamnum, &Ku1, &Kv1, 0);

                        Vec3f ptp = host->targetPrecalc[target->idx].PRE_RTll *
                                        Vec3f(xs/SphereRadius, ys/SphereRadius, zs/SphereRadius) / p->idepth_scaled + host->targetPrecalc[target->idx].PRE_tTll;    // projected point with real depth.
                        S_norm_pt = SphereRadius/ptp.norm();

                        float xs2 = S_norm_pt*ptp(0);
                        float ys2 = S_norm_pt*ptp(1);
                        float zs2 = S_norm_pt*ptp(2);

                        // double Ku2, Kv2;
                        // UMF->LadybugReprojectSpherePtToRectifyfixNum(xs, ys, zs, tocamnum, &Ku2, &Kv2, 0);
                        
                        float relBS = 0.01 * (Vec3f(xs1 - xs2, ys1 - ys2, zs1 - zs2)).norm();    // 0.01 = one pixel.    
                        if (relBS > p->maxRelBaseline)
                            p->maxRelBaseline = relBS;     // 正比于球面点的基线长度

                        p->numGoodResiduals++;
                    }
                } 
                else 
                {
                    toRemove[tid].push_back(activeResiduals[k]);  // // 删除OOB, Outlier 残差太大则移除
                }
            }
        }
    }

    // applies step to linearization point.
    bool FullSystem::doStepFromBackup(float stepfacC, float stepfacT, float stepfacR, float stepfacA, float stepfacD) {

        Vec10 pstepfac;
        pstepfac.segment<3>(0).setConstant(stepfacT);
        pstepfac.segment<3>(3).setConstant(stepfacR);
        pstepfac.segment<4>(6).setConstant(stepfacA);

        float sumA = 0, sumB = 0, sumT = 0, sumR = 0, sumID = 0, numID = 0;

        float sumNID = 0;

        if (setting_solverMode & SOLVER_MOMENTUM) {
            Hcalib->mpCH->setValue(Hcalib->mpCH->value_backup + Hcalib->mpCH->step);
            for (auto &fr:frames) {
                auto fh = fr->frameHessian;
                Vec10 step = fh->step;
                step.head<6>() += 0.5f * (fh->step_backup.head<6>());

                fh->setState(fh->state_backup + step);
                sumA += step[6] * step[6];
                sumB += step[7] * step[7];
                sumT += step.segment<3>(0).squaredNorm();
                sumR += step.segment<3>(3).squaredNorm();

                for (auto feat: fr->features) {
                    if (feat->status == Feature::FeatureStatus::VALID
                        && feat->point && feat->point->status == Point::PointStatus::ACTIVE) {

                        auto ph = feat->point->mpPH;
                        float step = ph->step + 0.5f * (ph->step_backup);
                        ph->setIdepth(ph->idepth_backup + step);
                        sumID += step * step;
                        sumNID += fabsf(ph->idepth_backup);
                        numID++;
                        ph->setIdepthZero(ph->idepth_backup + step);
                    }
                }
            }
        } else {
            Hcalib->mpCH->setValue(Hcalib->mpCH->value_backup + stepfacC * Hcalib->mpCH->step);
            for (auto &fr: frames) {
                auto fh = fr->frameHessian;
                fh->setState(fh->state_backup + pstepfac.cwiseProduct(fh->step));
                sumA += fh->step[6] * fh->step[6];
                sumB += fh->step[7] * fh->step[7];
                sumT += fh->step.segment<3>(0).squaredNorm();
                sumR += fh->step.segment<3>(3).squaredNorm();

                for (auto feat: fr->features) {
                    if (feat->status == Feature::FeatureStatus::VALID && feat->point &&
                        feat->point->status == Point::PointStatus::ACTIVE) {
                        auto ph = feat->point->mpPH;
                        ph->setIdepth(ph->idepth_backup + stepfacD * ph->step);
                        sumID += ph->step * ph->step;
                        sumNID += fabsf(ph->idepth_backup);
                        numID++;
                        ph->setIdepthZero(ph->idepth_backup + stepfacD * ph->step);

                    }
                }
            }
        }

        sumA /= frames.size();
        sumB /= frames.size();
        sumR /= frames.size();
        sumT /= frames.size();
        sumID /= numID;
        sumNID /= numID;

        EFDeltaValid = false;
        setPrecalcValues();

        return sqrtf(sumA) < 0.0005 * setting_thOptIterations &&
               sqrtf(sumB) < 0.00005 * setting_thOptIterations &&
               sqrtf(sumR) < 0.00005 * setting_thOptIterations &&
               sqrtf(sumT) * sumNID < 0.00005 * setting_thOptIterations;
    }

    // for multi-fisheye
    bool FullSystem::doStepFromBackupforMF(float stepfacT, float stepfacR, float stepfacA, float stepfacD)
    {
        // 相当于步长
        Vec16 pstepfac;
        pstepfac.segment<3>(0).setConstant(stepfacT);
        pstepfac.segment<3>(3).setConstant(stepfacR);
        pstepfac.segment<10>(6).setConstant(stepfacA);

        float sumA = 0, sumB = 0, sumT = 0, sumR = 0, sumID = 0, numID = 0;

        float sumNID = 0;

        if (setting_solverMode & SOLVER_MOMENTUM) 
        {
            for (auto &fr:frames) 
            {
                auto fh = fr->frameHessian;
                Vec16 step = fh->stepforMF;
                step.head<6>() += 0.5f * (fh->step_backupforMF.head<6>());

                fh->setStateforMF(fh->state_backupforMF + step);
                for(int n = 0; n < UMF->camNums; n++)
                {
                    sumA += step[6 + 2*n] * step[6 + 2*n];
                    sumB += step[7 + 2*n] * step[7 + 2*n];
                }
                sumA = sumA / UMF->camNums;         // // 光度增量平方
                sumB = sumB / UMF->camNums;
                sumT += step.segment<3>(0).squaredNorm();   // 平移增量
                sumR += step.segment<3>(3).squaredNorm();   // 旋转增量

                for (auto feat: fr->features) 
                {
                    if (feat->status == Feature::FeatureStatus::VALID
                        && feat->point && feat->point->status == Point::PointStatus::ACTIVE) 
                    {
                        auto ph = feat->point->mpPH;
                        float step = ph->step + 0.5f * (ph->step_backup);
                        ph->setIdepth(ph->idepth_backup + step);
                        sumID += step * step;
                        sumNID += fabsf(ph->idepth_backup);
                        numID++;
                        ph->setIdepthZero(ph->idepth_backup + step);
                    }
                }
            }
        } 
        else 
        {
            for (auto &fr: frames) 
            {
                auto fh = fr->frameHessian;
                fh->stepforMF.tail<10>() = Vec10::Zero(); //不更新光度系数
                fh->setStateforMF(fh->state_backupforMF + pstepfac.cwiseProduct(fh->stepforMF));
                for(int c = 0; c < UMF->camNums; c++)
                {
                    sumA += fh->stepforMF[6 + c * 2] * fh->stepforMF[6 + c * 2];
                    sumB += fh->stepforMF[7 + c * 2] * fh->stepforMF[7 + c * 2];
                }
                sumA = sumA / UMF->camNums;
                sumB = sumB / UMF->camNums;
                sumT += fh->stepforMF.segment<3>(0).squaredNorm();
                sumR += fh->stepforMF.segment<3>(3).squaredNorm();

                for (auto feat: fr->features) 
                {
                    if (feat->status == Feature::FeatureStatus::VALID && feat->point &&
                        feat->point->status == Point::PointStatus::ACTIVE) 
                    {
                        auto ph = feat->point->mpPH;
                        ph->setIdepth(ph->idepth_backup + stepfacD * ph->step);
                        sumID += ph->step * ph->step;
                        sumNID += fabsf(ph->idepth_backup);
                        numID++;
                        ph->setIdepthZero(ph->idepth_backup + stepfacD * ph->step);

                    }
                }
            }
        }

        sumA /= frames.size();
        sumB /= frames.size();
        sumR /= frames.size();
        sumT /= frames.size();
        sumID /= numID;
        sumNID /= numID;

        EFDeltaValid = false;
        setPrecalcValuesforMF();    // 更新相对位姿, 光度

        // 步长小于阈值则可以停止了
        return sqrtf(sumA) < 0.0005 * setting_thOptIterations &&
               sqrtf(sumB) < 0.00005 * setting_thOptIterations &&
               sqrtf(sumR) < 0.00005 * setting_thOptIterations &&
               sqrtf(sumT) * sumNID < 0.00005 * setting_thOptIterations;
    }

    void FullSystem::backupState(bool backupLastStep) {

        if (setting_solverMode & SOLVER_MOMENTUM) {
            if (backupLastStep) {
                Hcalib->mpCH->step_backup = Hcalib->mpCH->step;
                Hcalib->mpCH->value_backup = Hcalib->mpCH->value;
                for (auto &fr: frames) {
                    auto fh = fr->frameHessian;
                    fh->step_backup = fh->step;
                    fh->state_backup = fh->get_state();
                    for (auto feat: fr->features) {
                        if (feat->point && feat->point->status == Point::PointStatus::ACTIVE) {
                            auto ph = feat->point->mpPH;
                            ph->idepth_backup = ph->idepth;
                            ph->step_backup = ph->step;
                        }
                    }
                }
            } else {
                Hcalib->mpCH->step_backup.setZero();
                Hcalib->mpCH->value_backup = Hcalib->mpCH->value;
                for (auto &fr: frames) {
                    auto fh = fr->frameHessian;
                    fh->step_backup.setZero();
                    fh->state_backup = fh->get_state();
                    for (auto feat: fr->features) {
                        if (feat->point && feat->point->status == Point::PointStatus::ACTIVE) {
                            auto ph = feat->point->mpPH;
                            ph->idepth_backup = ph->idepth;
                            ph->step_backup = 0;
                        }
                    }
                }
            }
        } else {
            Hcalib->mpCH->value_backup = Hcalib->mpCH->value;
            for (auto &fr: frames) {
                auto fh = fr->frameHessian;
                fh->state_backup = fh->get_state();
                for (auto feat: fr->features) {
                    if (feat->status == Feature::FeatureStatus::VALID &&
                        feat->point->status == Point::PointStatus::ACTIVE) {
                        auto ph = feat->point->mpPH;
                        ph->idepth_backup = ph->idepth;
                    }
                }
            }
        }
    }

    // for multi-fisheye  对帧, 点, 内参的step和state进行备份
    void FullSystem::backupStateforMF(bool backupLastStep)
    {
        if (setting_solverMode & SOLVER_MOMENTUM) 
        {
            if (backupLastStep) 
            {
                for (auto &fr: frames) 
                {
                    auto fh = fr->frameHessian;
                    fh->step_backupforMF = fh->stepforMF;
                    fh->state_backupforMF = fh->get_stateforMF();
                    for (auto feat: fr->features) 
                    {
                        if (feat->point && feat->point->status == Point::PointStatus::ACTIVE) 
                        {
                            auto ph = feat->point->mpPH;
                            ph->idepth_backup = ph->idepth;
                            ph->step_backup = ph->step;
                        }
                    }
                }
            } 
            else    // 迭代前初始化
            {
                for (auto &fr: frames) 
                {
                    auto fh = fr->frameHessian;
                    fh->step_backupforMF.setZero();
                    fh->state_backupforMF = fh->get_stateforMF();
                    for (auto feat: fr->features) 
                    {
                        if (feat->point && feat->point->status == Point::PointStatus::ACTIVE) 
                        {
                            auto ph = feat->point->mpPH;
                            ph->idepth_backup = ph->idepth;
                            ph->step_backup = 0;
                        }
                    }
                }
            }
        } 
        else 
        {
            for (auto &fr: frames) 
            {
                auto fh = fr->frameHessian;
                fh->state_backupforMF = fh->get_stateforMF();
                for (auto feat: fr->features) 
                {
                    if (feat->status == Feature::FeatureStatus::VALID &&
                        feat->point->status == Point::PointStatus::ACTIVE) 
                    {
                        auto ph = feat->point->mpPH;
                        ph->idepth_backup = ph->idepth;
                    }
                }
            }
        }
    }

    void FullSystem::loadSateBackup() {

        Hcalib->mpCH->setValue(Hcalib->mpCH->value_backup);
        for (auto fr: frames) {
            auto fh = fr->frameHessian;
            fh->setState(fh->state_backup);
            for (auto feat: fr->features) {
                if (feat->point && feat->point->status == Point::PointStatus::ACTIVE) {
                    auto ph = feat->point->mpPH;
                    ph->setIdepth(ph->idepth_backup);
                    ph->setIdepthZero(ph->idepth_backup);
                }
            }
        }

        EFDeltaValid = false;
        setPrecalcValues();
    }

    // for multi-fisheye
    void FullSystem::loadSateBackupforMF() 
    {
        for (auto fr: frames) {
            auto fh = fr->frameHessian;
            fh->setStateforMF(fh->state_backupforMF);
            for (auto feat: fr->features) {
                if (feat->point && feat->point->status == Point::PointStatus::ACTIVE) {

                    auto ph = feat->point->mpPH;
                    ph->setIdepth(ph->idepth_backup);
                    ph->setIdepthZero(ph->idepth_backup);
                }
            }
        }

        EFDeltaValid = false;
        setPrecalcValuesforMF();
    }

    double FullSystem::calcLEnergy() {
        if (setting_forceAceptStep)
            return 0;
        return ef->calcLEnergyF_MT();
    }

    double FullSystem::calcMEnergy() {
        if (setting_forceAceptStep)
            return 0;
        return ef->calcMEnergyF();
    }

    // // for multi-fisheye
    double FullSystem::calcLEnergyforMF() {
        if (setting_forceAceptStep)
            return 0;
        return ef->calcLEnergyF_MTforMF();
    }
    // for multi-fisheye
    double FullSystem::calcMEnergyforMF() {
        if (setting_forceAceptStep)
            return 0;
        return ef->calcMEnergyFforMF();
    }

    void FullSystem::applyRes_Reductor(bool copyJacobians, int min, int max, Vec10 *stats, int tid) {
        for (int k = min; k < max; k++)
            activeResiduals[k]->applyRes(true);
    }

    std::vector<VecX> FullSystem::getNullspaces(std::vector<VecX> &nullspaces_pose, std::vector<VecX> &nullspaces_scale,
                                                std::vector<VecX> &nullspaces_affA,
                                                std::vector<VecX> &nullspaces_affB) {

        nullspaces_pose.clear();
        nullspaces_scale.clear();
        nullspaces_affA.clear();
        nullspaces_affB.clear();

        int n = CPARS + frames.size() * 8;
        std::vector<VecX> nullspaces_x0_pre;
        for (int i = 0; i < 6; i++) {
            VecX nullspace_x0(n);
            nullspace_x0.setZero();
            for (auto fr: frames) {
                auto fh = fr->frameHessian;
                nullspace_x0.segment<6>(CPARS + fh->idx * 8) = fh->nullspaces_pose.col(i);
                nullspace_x0.segment<3>(CPARS + fh->idx * 8) *= SCALE_XI_TRANS_INVERSE;
                nullspace_x0.segment<3>(CPARS + fh->idx * 8 + 3) *= SCALE_XI_ROT_INVERSE;
            }
            nullspaces_x0_pre.push_back(nullspace_x0);
            nullspaces_pose.push_back(nullspace_x0);
        }
        for (int i = 0; i < 2; i++) {
            VecX nullspace_x0(n);
            nullspace_x0.setZero();
            for (auto fr: frames) {
                auto fh = fr->frameHessian;
                nullspace_x0.segment<2>(CPARS + fh->idx * 8 + 6) = fh->nullspaces_affine.col(i).head<2>();
                nullspace_x0[CPARS + fh->idx * 8 + 6] *= SCALE_A_INVERSE;
                nullspace_x0[CPARS + fh->idx * 8 + 7] *= SCALE_B_INVERSE;
            }
            nullspaces_x0_pre.push_back(nullspace_x0);
            if (i == 0) nullspaces_affA.push_back(nullspace_x0);
            if (i == 1) nullspaces_affB.push_back(nullspace_x0);
        }

        VecX nullspace_x0(n);
        nullspace_x0.setZero();
        for (auto fr: frames) {
            auto fh = fr->frameHessian;
            nullspace_x0.segment<6>(CPARS + fh->idx * 8) = fh->nullspaces_scale;
            nullspace_x0.segment<3>(CPARS + fh->idx * 8) *= SCALE_XI_TRANS_INVERSE;
            nullspace_x0.segment<3>(CPARS + fh->idx * 8 + 3) *= SCALE_XI_ROT_INVERSE;
        }
        nullspaces_x0_pre.push_back(nullspace_x0);
        nullspaces_scale.push_back(nullspace_x0);

        return nullspaces_x0_pre;
    }

    std::vector<VecX> FullSystem::getNullspacesforMF(std::vector<VecX> &nullspaces_pose, std::vector<VecX> &nullspaces_scale) 
    {

        nullspaces_pose.clear();
        nullspaces_scale.clear();
        // nullspaces_affA.clear();
        // nullspaces_affB.clear();

        int n = frames.size() * 16;
        std::vector<VecX> nullspaces_x0_pre;
        // pose
        for (int i = 0; i < 6; i++) 
        {
            VecX nullspace_x0(n);
            nullspace_x0.setZero();
            for (auto fr: frames) 
            {
                auto fh = fr->frameHessian;
                nullspace_x0.segment<6>(fh->idx * 16) = fh->nullspaces_pose.col(i);
                nullspace_x0.segment<3>(fh->idx * 16) *= SCALE_XI_TRANS_INVERSE;
                nullspace_x0.segment<3>(fh->idx * 16 + 3) *= SCALE_XI_ROT_INVERSE;
            }
            nullspaces_x0_pre.push_back(nullspace_x0);
            nullspaces_pose.push_back(nullspace_x0);
        }
        // // Light affine ab
        // for (int i = 0; i < 2; i++) 
        // {
        //     VecX nullspace_x0(n);
        //     nullspace_x0.setZero();
        //     for (auto fr: frames) 
        //     {
        //         auto fh = fr->frameHessian;
        //         nullspace_x0.segment<2>(fh->idx * 16 + 6) = fh->nullspaces_affine.col(i).head<2>();
        //         nullspace_x0[fh->idx * 16 + 6] *= SCALE_A_INVERSE;
        //         nullspace_x0[fh->idx * 16 + 7] *= SCALE_B_INVERSE;
        //     }
        //     nullspaces_x0_pre.push_back(nullspace_x0);
        //     if (i == 0) nullspaces_affA.push_back(nullspace_x0);
        //     if (i == 1) nullspaces_affB.push_back(nullspace_x0);
        // }

        VecX nullspace_x0(n);
        nullspace_x0.setZero();
        for (auto fr: frames) {
            auto fh = fr->frameHessian;
            nullspace_x0.segment<6>(fh->idx * 16) = fh->nullspaces_scale;
            nullspace_x0.segment<3>(fh->idx * 16) *= SCALE_XI_TRANS_INVERSE;
            nullspace_x0.segment<3>(fh->idx * 16 + 3) *= SCALE_XI_ROT_INVERSE;
        }
        nullspaces_x0_pre.push_back(nullspace_x0);
        nullspaces_scale.push_back(nullspace_x0);

        return nullspaces_x0_pre;
    }

    void FullSystem::setNewFrameEnergyTH() {

        // collect all residuals and make decision on TH.
        allResVec.clear();
        allResVec.reserve(activeResiduals.size() * 2);
        auto fr = frames.back();
        auto newFrame = fr->frameHessian;

        for (auto &r : activeResiduals)
            if (r->state_NewEnergyWithOutlier >= 0 && r->target.lock() == newFrame) {
                allResVec.push_back(r->state_NewEnergyWithOutlier);
            }

        if (allResVec.size() == 0) {
            newFrame->frameEnergyTH = 12 * 12 * patternNum;
            return;        // should never happen, but lets make sure.
        }

        int nthIdx = setting_frameEnergyTHN * allResVec.size();

        assert(nthIdx < (int) allResVec.size());
        assert(setting_frameEnergyTHN < 1);

        std::nth_element(allResVec.begin(), allResVec.begin() + nthIdx, allResVec.end());
        float nthElement = sqrtf(allResVec[nthIdx]);

        newFrame->frameEnergyTH = nthElement * setting_frameEnergyTHFacMedian;
        newFrame->frameEnergyTH = 26.0f * setting_frameEnergyTHConstWeight +
                                  newFrame->frameEnergyTH * (1 - setting_frameEnergyTHConstWeight);
        newFrame->frameEnergyTH = newFrame->frameEnergyTH * newFrame->frameEnergyTH;
        newFrame->frameEnergyTH *= setting_overallEnergyTHWeight * setting_overallEnergyTHWeight;
    }

    void FullSystem::printOptRes(const Vec3 &res, double resL, double resM, double resPrior, double LExact, float a,
                                 float b) {
        char buff[256] = {};
        sprintf(buff, "A(%f)=(AV %.3f). Num: A(%'d) + M(%'d); ab %f %f!\n",
                res[0],
                sqrtf((float) (res[0] / (patternNum * ef->resInA))),
                ef->resInA,
                ef->resInM,
                a,
                b
        );
        LOG(INFO) << string(buff);
    }

    void FullSystem::printOptResforMF(const Vec3 &res, double resL, double resM, double resPrior, double LExact, vector<AffLight> va)
    {
        char buff[256] = {};
        sprintf(buff, "A(%f)=(AV %.3f). Num: A(%'d) + M(%'d); ab_0 %f %f; ab_1 %f %f; ab_2 %f %f; ab_3 %f %f; ab_4 %f %f!\n",
                res[0],
                sqrtf((float) (res[0] / (patternNum * ef->resInA))),
                ef->resInA,
                ef->resInM,
                va[0].a,
                va[0].b,
                va[1].a,
                va[1].b,
                va[2].a,
                va[2].b,
                va[3].a,
                va[3].b,
                va[4].a,
                va[4].b
        );
        LOG(INFO) << string(buff);
    }

    

    void FullSystem::mappingLoop() {

        unique_lock<mutex> lock(trackMapSyncMutex);

        while (runMapping) {

            // wait an unmapped frame
            while (unmappedTrackedFrames.size() == 0) {
                trackedFrameSignal.wait(lock);
                if (!runMapping) break;
            }
            if (!runMapping) break;

            // get an unmapped frame, tackle it.
            shared_ptr<Frame> fr = unmappedTrackedFrames.front();
            auto fh = fr->frameHessian;
            unmappedTrackedFrames.pop_front();

            // guaranteed to make a KF for the very first two tracked frames.
            if (globalMap->NumFrames() <= 2) {
                lock.unlock();
                makeKeyFrame(fh);
                lock.lock();
                mappedFrameSignal.notify_all();
                continue;
            }

            if (unmappedTrackedFrames.size() > 3)
                needToKetchupMapping = true;

            if (unmappedTrackedFrames.size() > 0) {
                // if there are other frames to track, do that first.
                lock.unlock();
                makeNonKeyFrame(fh);
                lock.lock();

                if (needToKetchupMapping && unmappedTrackedFrames.size() > 0) {
                    auto fr = unmappedTrackedFrames.front();
                    shared_ptr<FrameHessian> fh = fr->frameHessian;
                    unmappedTrackedFrames.pop_front();
                    {
                        unique_lock<mutex> crlock(shellPoseMutex);
                        fh->setEvalPT_scaled(fr->getPose(), fh->frame->aff_g2l);
                    }
                }
            } else {
                if (setting_realTimeMaxKF || needNewKFAfter >= int(frames.back()->id)) {
                    lock.unlock();
                    makeKeyFrame(fh);
                    needToKetchupMapping = false;
                    lock.lock();
                } else {
                    lock.unlock();
                    makeNonKeyFrame(fh);
                    lock.lock();
                }
            }

            mappedFrameSignal.notify_all();
        }
        LOG(INFO) << "MAPPING FINISHED!";
    }

    void FullSystem::mappingLoopforMF() 
    {

        unique_lock<mutex> lock(trackMapSyncMutex);

        while (runMapping) {

            // wait an unmapped frame
            while (unmappedTrackedFrames.size() == 0) {
                trackedFrameSignal.wait(lock);
                if (!runMapping) break;
            }
            if (!runMapping) break;

            // get an unmapped frame, tackle it.
            shared_ptr<Frame> fr = unmappedTrackedFrames.front();
            auto fh = fr->frameHessian;
            unmappedTrackedFrames.pop_front();

            // guaranteed to make a KF for the very first two tracked frames.
            if (globalMap->NumFrames() <= 2) {
                lock.unlock();
                makeKeyFrameforMF(fh);
                lock.lock();
                mappedFrameSignal.notify_all();
                continue;
            }

            if (unmappedTrackedFrames.size() > 3)
                needToKetchupMapping = true;

            if (unmappedTrackedFrames.size() > 0) {
                // if there are other frames to track, do that first.
                lock.unlock();
                makeNonKeyFrameforMF(fh);
                lock.lock();

                if (needToKetchupMapping && unmappedTrackedFrames.size() > 0) {
                    auto fr = unmappedTrackedFrames.front();
                    shared_ptr<FrameHessian> fh = fr->frameHessian;
                    unmappedTrackedFrames.pop_front();
                    {
                        unique_lock<mutex> crlock(shellPoseMutex);
                        fh->setEvalPT_scaled(fr->getPose(), fh->frame->aff_g2l);
                    }
                }
            } else {
                if (setting_realTimeMaxKF || needNewKFAfter >= int(frames.back()->id)) {
                    lock.unlock();
                    makeKeyFrameforMF(fh);
                    needToKetchupMapping = false;
                    lock.lock();
                } else {
                    lock.unlock();
                    makeNonKeyFrameforMF(fh);
                    lock.lock();
                }
            }

            mappedFrameSignal.notify_all();
        }
        LOG(INFO) << "MAPPING FINISHED!";
    }

    bool FullSystem::saveAll(const string &filename) {
        ofstream fout(filename, ios::out | ios::binary);
        if (!fout) return false;
        int nKF = globalMap->NumFrames();
        fout.write((char *) &nKF, sizeof(int));
        auto allKFs = globalMap->GetAllKFs();
        for (auto &frame: allKFs) {
            frame->save(fout);
        }
        fout.close();
        LOG(INFO) << "DONE!" << endl;
        return true;
    }

    bool FullSystem::loadAll(const string &filename) {

        ifstream fin(filename);
        if (!fin) return false;
        int numKF = 0;
        fin.read((char *) &numKF, sizeof(numKF));

        vector<shared_ptr<Frame>> allKFs;
        allKFs.resize(numKF, nullptr);
        for (auto &kf: allKFs) {
            kf = shared_ptr<Frame>(new Frame());
        }

        int i = 0;
        while (!fin.eof() && i < int(allKFs.size())) {
            shared_ptr<Frame> &newFrame = allKFs[i];
            newFrame->load(fin, newFrame, allKFs);
            i++;
        }

        fin.close();

        if (viewer)
            viewer->publishKeyframes(allKFs, false, Hcalib->mpCH);

        frames = allKFs;
        for (auto &kf: allKFs) {
            globalMap->AddKeyFrame(kf);
        }

        LOG(INFO) << "Loaded total " << frames.size() << " keyframes" << endl;
        return true;
    }

    void FullSystem::printResult(const string &filename, bool printOptimized) {

        unique_lock<mutex> lock(trackMutex);
        unique_lock<mutex> crlock(shellPoseMutex);

        std::ofstream myfile(filename);
        myfile << std::setprecision(15);

        auto allKFs = globalMap->GetAllKFs();
        LOG(INFO) << "total keyframes: " << allKFs.size() << endl;

        for (auto &fr : allKFs) {
            SE3 Twc;
            Sim3 Swc;
            if (printOptimized) {
                Swc = fr->getPoseOpti().inverse();
                Twc = SE3(Swc.rotationMatrix(), Swc.translation());
            } else
                Twc = fr->getPose().inverse();

            myfile << fr->timeStamp <<
                   " " << Twc.translation().transpose() <<
                   " " << Twc.so3().unit_quaternion().x() <<
                   " " << Twc.so3().unit_quaternion().y() <<
                   " " << Twc.so3().unit_quaternion().z() <<
                   " " << Twc.so3().unit_quaternion().w() << "\n";
        }
        myfile.close();
    }

    void FullSystem::printResultKitti(const string &filename, bool printOptimized) {

        LOG(INFO) << "saving kitti trajectory..." << endl;

        unique_lock<mutex> lock(trackMutex);
        unique_lock<mutex> crlock(shellPoseMutex);
        ofstream f(filename);

        auto allKFs = globalMap->GetAllKFs();
        for (auto &fr: allKFs) {
            if (printOptimized == false) {
                SE3 Twc = fr->getPose().inverse();
                Mat33 Rwc = Twc.rotationMatrix();
                Vec3 twc = Twc.translation();
                f << fr->id << " " << setprecision(9) <<
                  Rwc(0, 0) << " " << Rwc(0, 1) << " " << Rwc(0, 2) << " " << twc(0) << " " <<
                  Rwc(1, 0) << " " << Rwc(1, 1) << " " << Rwc(1, 2) << " " << twc(1) << " " <<
                  Rwc(2, 0) << " " << Rwc(2, 1) << " " << Rwc(2, 2) << " " << twc(2) << endl;
            } else {
                Sim3 Swc = fr->getPoseOpti().inverse();
                Mat33 Rwc = Swc.rotationMatrix();
                Vec3 twc = Swc.translation();
                f << fr->id << " " << setprecision(9) <<
                  Rwc(0, 0) << " " << Rwc(0, 1) << " " << Rwc(0, 2) << " " << twc(0) << " " <<
                  Rwc(1, 0) << " " << Rwc(1, 1) << " " << Rwc(1, 2) << " " << twc(1) << " " <<
                  Rwc(2, 0) << " " << Rwc(2, 1) << " " << Rwc(2, 2) << " " << twc(2) << endl;
            }
        }
        f.close();

        LOG(INFO) << "done." << endl;
    }
}
