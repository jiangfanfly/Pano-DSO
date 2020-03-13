#pragma once
#ifndef LDSO_COARSE_TRACKER_H_
#define LDSO_COARSE_TRACKER_H_

#include "NumTypes.h"
#include "internal/OptimizationBackend/MatrixAccumulators.h"
#include "internal/Residuals.h"
#include "internal/FrameHessian.h"
#include "internal/CalibHessian.h"
#include "frontend/Undistort.h"
#include "DSOViewer.h"

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

using namespace ldso;
using namespace ldso::internal;

namespace ldso {

    // the tracker
    class CoarseTracker {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        // constuctor, allocate memory and compute the camera intrinsics pyramid
        CoarseTracker(int w, int h);

        ~CoarseTracker() {
            for (float *ptr : ptrToDelete)
                delete[] ptr;
            ptrToDelete.clear();
        }

        /**
         ** @brief track the new coming frame and estimate its pose and light parameters
         * @param[in] newFrameHessian the new frame
         * @param[in] lastToNew_out initial value of T_new_last
         * @param[out] aff_g2l_out affine transform
         * @param[in] coarsestLvl the first pyramid level (default=5)
         * @param[in] minResForAbort if residual > 1.5* minResForAbort, return false
         * @return true if track is good
         */
        bool trackNewestCoarse(
                shared_ptr<FrameHessian> newFrameHessian,
                SE3 &lastToNew_out, AffLight &aff_g2l_out,
                int coarsestLvl, Vec5 minResForAbort);

        void debugPlotIDepthMap(float* minID_pt, float* maxID_pt, shared_ptr<PangolinDSOViewer> wraps);

        void setCoarseTrackingRef(
                std::vector<shared_ptr<FrameHessian>>& frameHessians);

        /**
         * Create camera intrinsics buffer given the calibrated parameters
         * @param HCalib
         */
        void makeK(
                shared_ptr<CalibHessian> HCalib);

        shared_ptr<FrameHessian> lastRef = nullptr;     // the reference frame
        AffLight lastRef_aff_g2l;                       // affine light transform
        shared_ptr<FrameHessian> newFrame = nullptr;    // the new coming frame
        int refFrameID = -1;

        // act as pure ouptut
        Vec5 lastResiduals;
        Vec3 lastFlowIndicators;
        double firstCoarseRMSE = 0;

        // camera and image parameters in each pyramid
        Mat33f K[PYR_LEVELS];
        Mat33f Ki[PYR_LEVELS];
        float fx[PYR_LEVELS];
        float fy[PYR_LEVELS];
        float fxi[PYR_LEVELS];
        float fyi[PYR_LEVELS];
        float cx[PYR_LEVELS];
        float cy[PYR_LEVELS];
        float cxi[PYR_LEVELS];
        float cyi[PYR_LEVELS];
        int w[PYR_LEVELS];
        int h[PYR_LEVELS];

        // ===========================
        // for multi-fisheye
        CoarseTracker(UndistortMultiFisheye* uMF);

        void debugPlotIDepthMapforMF(float* minID_pt, float* maxID_pt, shared_ptr<PangolinDSOViewer> wraps);

        void setCoarseTrackingRefforMF(std::vector<shared_ptr<FrameHessian>>& frameHessians);

        /**
         ** @brief track the new coming frame and estimate its pose and light parameters
         * @param[in] newFrameHessian the new frame
         * @param[in] lastToNew_out initial value of T_new_last
         * @param[out] aff_g2l_out affine transform
         * @param[in] coarsestLvl the first pyramid level (default=5)
         * @param[in] minResForAbort if residual > 1.5* minResForAbort, return false
         * @return true if track is good
         */
        bool trackNewestCoarseforMF(
                shared_ptr<FrameHessian> newFrameHessian,
                SE3 &lastToNew_out, vector<AffLight> &vaff_g2l_out,
                int coarsestLvl, Vec5 minResForAbort);

        UndistortMultiFisheye* UMF;

        vector<AffLight> vlastRef_aff_g2l;                       // affine light transform

    private:
        void makeCoarseDepthL0(std::vector<shared_ptr<FrameHessian>> frameHessians);

        /**
         * @param[in] lvl the pyramid level
         * @param[in] refToNew pose from reference to current
         * @param[in] aff_g2l affine light transform from g to l
         * @param[in] cutoffTH cut off threshold, if residual > cutoffTH, then make residual = max energy. Similar with robust kernel in g2o.
         * @return the residual vector (a bit complicated, the the last lines in this func.)
         */
        Vec6 calcRes(int lvl, const SE3 &refToNew, AffLight aff_g2l, float cutoffTH);

        /**
         * @brief SSE accelerated Gauss-Newton
         * NOTE it uses some cache data in "warped buffers"
         * @param[in] lvl image pyramid level
         * @param[out] H_out Hessian matrix
         * @param[out] b_out bias vector
         * @param[in] refToNew Transform matrix from ref to new
         * @param[in] aff_g2l affine light transform
         */
        void calcGSSSE(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l);


        // point cloud buffers
        // wxh in each pyramid layer
        float *pc_u[PYR_LEVELS];            // u coordinates
        float *pc_v[PYR_LEVELS];            // v coordinates
        float *pc_idepth[PYR_LEVELS];       // inv depth in the reference
        float *pc_color[PYR_LEVELS];        // color of the reference patches
        int pc_n[PYR_LEVELS];               // number of points in each layer

        // warped buffer, used as wxh images
        float *buf_warped_idepth;
        float *buf_warped_u;
        float *buf_warped_v;
        float *buf_warped_dx;
        float *buf_warped_dy;
        float *buf_warped_residual;
        float *buf_warped_weight;
        float *buf_warped_refColor;
        int buf_warped_n;

        float *idepth[PYR_LEVELS];
        float *weightSums[PYR_LEVELS];
        float *weightSums_bak[PYR_LEVELS];

        std::vector<float *> ptrToDelete;    // all allocated memory, will be deleted in deconstructor
        std::vector<int *> ptrToDeletei; 
        Accumulator9 acc;

        // =========================
        // for multi-fisheye
        void makeCoarseDepthL0forMF(std::vector<shared_ptr<FrameHessian>> frameHessians);

        /**
         * @param[in] lvl the pyramid level
         * @param[in] refToNew pose from reference to current
         * @param[in] aff_g2l affine light transform from g to l
         * @param[in] cutoffTH cut off threshold, if residual > cutoffTH, then make residual = max energy. Similar with robust kernel in g2o.
         * @return the residual vector (a bit complicated, the the last lines in this func.)
         */
        Vec6 calcResforMF(int lvl, const SE3 &refToNew, vector<vector<Vec2f>> vvaff, float cutoffTH);

        /**
         * @brief SSE accelerated Gauss-Newton
         * NOTE it uses some cache data in "warped buffers"
         * @param[in] lvl image pyramid level
         * @param[out] H_out Hessian matrix
         * @param[out] b_out bias vector
         * @param[in] refToNew Transform matrix from ref to new
         * @param[in] aff_g2l affine light transform
         */
        void calcGSSSEforMF(int lvl, Mat5656 &H_out, Vec56 &b_out, const SE3 &refToNew);

        inline void computedxrdposedrho1(const Mat33f R, const Vec3f t, const Vec3f XcF, const Vec3f Xc, const int n, const float rho1, Eigen::Matrix<float, 2, 1> &dxrdrho1, Eigen::Matrix<float, 2, 6> &dxrdpose, int level) 
        {
            Eigen::Matrix2f dIL;
            dIL << 0,-1,1,0;

            float SphereRadius = UMF->GetSphereRadius();
            float Xcnorm_S = SphereRadius / Xc.norm();
            float xs = Xc(0) * Xcnorm_S;
            float ys = Xc(1) * Xcnorm_S;
            float zs = Xc(2) * Xcnorm_S;

            Eigen::Matrix<float, 2, 3> duXs;

            Eigen::Matrix<float, 3, 1> RaySphere;
            RaySphere(0, 0) = xs;
            RaySphere(1, 0) = ys;
            RaySphere(2, 0) = zs;

            Eigen::Matrix<float, 3, 3> RotationMat = (UMF->mvExParas[n].block<3, 3>(0, 0)).cast<float>();
            Eigen::Matrix<float, 3, 1> CamPos = UMF->mvExParas[n].block<3, 1>(0, 3).cast<float>();
            Eigen::Matrix<float, 3, 1> RayCoor = RaySphere - CamPos;
            Eigen::Matrix<float, 3, 1> RectifiedImgPtCoor = RotationMat.inverse() * RayCoor;
            float xsn = RectifiedImgPtCoor(0, 0);
            float ysn = RectifiedImgPtCoor(1, 0);
            float zsn = RectifiedImgPtCoor(2, 0);

            float f = UMF->GetRectifiedImgFocalLength(n, level);

            duXs << f / zsn, 0, -xsn * f / (zsn * zsn),
                    0, f / zsn, -ysn * f / (zsn * zsn);
            
            Eigen::Matrix<float, 2, 3> dxrdXs;
            dxrdXs = dIL * duXs * RotationMat.inverse();

            float xc = Xc(0);
            float yc = Xc(1);
            float zc = Xc(2);
            float l2 = xc * xc + yc * yc + zc * zc;
            float l23 = l2 * sqrt(l2);

            Eigen::Matrix<float, 3, 3> dXsdXc;
            dXsdXc << SphereRadius  *(yc * yc + zc * zc) / l23, -xc * yc * SphereRadius / l23, -xc * zc * SphereRadius / l23,
                    -xc * yc * SphereRadius / l23, SphereRadius * (xc * xc + zc * zc) / l23, -yc * zc * SphereRadius / l23,
                    -xc * zc * SphereRadius / l23, -yc * zc * SphereRadius / l23, SphereRadius * (xc * xc + yc * yc) / l23;
            
            Eigen::Matrix<float, 3, 6> dXcdpose;
            dXcdpose << 1, 0, 0, 0, zc, -yc,
                        0, 1, 0, -zc, 0, xc,
                        0, 0, 1, yc, -xc, 0;
            
            Eigen::Matrix<float, 3, 1> dXcdrho1;
            // dXcdroh1 = t;
            dXcdrho1 = - R * XcF / SphereRadius / (rho1 * rho1);
            
            
            dxrdrho1 = dxrdXs * dXsdXc * dXcdrho1;
            dxrdpose = dxrdXs * dXsdXc * dXcdpose;
      
        }

        vector<float> NumericDiff(int n, double x, double y,int lvl, Mat33f R, Vec3f t, float idepth, Vec2f ab);
        Vec5f testJaccobian(int n, double RectifiedPixalx, double RectifiedPixaly, int lvl, Mat33f R, Vec3f t, 
                            float idepth, Vec2f ab);

        vector<float*> videpth[PYR_LEVELS];
        vector<float*> vweightSums[PYR_LEVELS];
        vector<float*> vweightSums_bak[PYR_LEVELS];

        // point cloud buffers
        // wxh in each pyramid layer
        vector<float*> vpc_u[PYR_LEVELS];            // u coordinates
        vector<float*> vpc_v[PYR_LEVELS];            // v coordinates
        vector<float*> vpc_idepth[PYR_LEVELS];       // inv depth in the reference
        vector<float*> vpc_color[PYR_LEVELS];        // color of the reference patches
        vector<int> vpc_n[PYR_LEVELS];               // number of points in each layer

        // warped buffer, used as wxh images
        //float *buf_warped_idepth;
        float *buf_warped_xsF;
        float *buf_warped_ysF;
        float *buf_warped_zsF;
        float *buf_warped_xsT;
        float *buf_warped_ysT;
        float *buf_warped_zsT;
        int* buf_warped_fromcamnum;
        int* buf_warped_tocamnum;
        float* buf_warped_a;
        float* buf_warped_b;
        float* buf_warped_b0;
        // float *buf_warped_dx;
        // float *buf_warped_dy;
        // float *buf_warped_residual;
        // float *buf_warped_weight;
        // float *buf_warped_refColor;
        // int buf_warped_n;

        Accumulator57 accforMF;

    };

    // the distance map
    class CoarseDistanceMap {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        CoarseDistanceMap(int w, int h);

        ~CoarseDistanceMap();

        void makeDistanceMap(
                std::vector<shared_ptr<FrameHessian>>& frameHessians,
                shared_ptr<FrameHessian> frame);

        void makeK(shared_ptr<CalibHessian> HCalib);

        float *fwdWarpedIDDistFinal;        //!< 距离场的数值

        Mat33f K[PYR_LEVELS];
        Mat33f Ki[PYR_LEVELS];
        float fx[PYR_LEVELS];
        float fy[PYR_LEVELS];
        float fxi[PYR_LEVELS];
        float fyi[PYR_LEVELS];
        float cx[PYR_LEVELS];
        float cy[PYR_LEVELS];
        float cxi[PYR_LEVELS];
        float cyi[PYR_LEVELS];
        int w[PYR_LEVELS];
        int h[PYR_LEVELS];

        void addIntoDistFinal(int u, int v);

        // for multi-fishey
        CoarseDistanceMap(UndistortMultiFisheye * uMF);

        vector<float *> vfwdWarpedIDDistFinal; 

        UndistortMultiFisheye* UMF;

        void makeDistanceMapforMF(
                std::vector<shared_ptr<FrameHessian>>& frameHessians,
                shared_ptr<FrameHessian> frame);

        void addIntoDistFinalforMF(int u, int v, int camnum);
        
    private:

        PointFrameResidual **coarseProjectionGrid;
        int *coarseProjectionGridNum;
        Eigen::Vector2i *bfsList1;      //!< 投影到frame的坐标
        Eigen::Vector2i *bfsList2;      //!< 和1轮换使用

        void growDistBFS(int bfsNum);   //生成每一层的距离, 第一层为1, 第二层为2....

        // for multi-fishey
        vector<Eigen::Vector2i *> vbfsList1;      //!< 投影到frame的坐标
        vector<Eigen::Vector2i *> vbfsList2;      //!< 和1轮换使用

        void growDistBFSforMF(vector<int> vbfsNum);
        void growDistBFSforMF2(int bfsNum, int camnum) ;

    };
}

#endif // LDSO_COARSE_TRACKER_H_
