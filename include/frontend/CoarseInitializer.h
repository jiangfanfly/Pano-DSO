#pragma once
#ifndef LDSO_COARSE_INITIALIZER_H_
#define LDSO_COARSE_INITIALIZER_H_

#include "NumTypes.h"
#include "Settings.h"
#include "AffLight.h"
#include "internal/OptimizationBackend/MatrixAccumulators.h"
#include "Undistort.h"
#include "DSOViewer.h"

#include "Camera.h"
#include "Frame.h"
#include "Point.h"

using namespace ldso;
using namespace ldso::internal;

namespace ldso {

    /**
     * point structure used in coarse initializer
     */
    struct Pnt {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        // index in jacobian. never changes (actually, there is no reason why).
        float u, v;

        float xs,ys,zs;

        // idepth / isgood / energy during optimization.
        float idepth;           //!< 该点对应参考帧的逆深度
        bool isGood;            //!< 点在新图像内, 相机前, 像素值有穷则好
        Vec2f energy;           //!< [0]残差的平方, [1]正则化项(逆深度减一的平方) // (UenergyPhotometric, energyRegularizer)
        bool isGood_new;
        float idepth_new;       //!< 该点对应参考帧的新的逆深度   
        Vec2f energy_new;       //!< 迭代计算的新的能量

        float iR;  // iR 变量相当于是逆深度的真值，在优化的过程中，使用这个值计算逆深度误差
        float iRSumNum;         //!< 子点逆深度信息矩阵之和

        float lastHessian;      //!< 逆深度的Hessian, 即协方差, dd*dd
        float lastHessian_new;  //!< 新一次迭代的协方差

        // max stepsize for idepth (corresponding to max. movement in pixel-space).
        float maxstep;          //!< 逆深度增加的最大步长

        // idx (x+y*w) of closest point one pyramid level above.
        int parent;             //!< 上一层中该点的父节点 (距离最近的)的id
        float parentDist;       //!< 上一层中与父节点的距离

        // idx (x+y*w) of up to 10 nearest points in pixel space.
        int neighbours[10];
        float neighboursDist[10];

        float my_type;          //!< 第0层提取是1, 2, 4, 对应d, 2d, 4d, 其它层是1
        float outlierTH;        //!< 外点阈值

        int tocamnum;           
        int hcamnum;
    };

    /**
     * initializer for monocular slam
     */
    class CoarseInitializer {
    public:

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        CoarseInitializer(int w, int h);

        CoarseInitializer(UndistortMultiFisheye* uMF);

        ~CoarseInitializer();


        void setFirst(shared_ptr<CalibHessian> HCalib, shared_ptr<FrameHessian> newFrameHessian);

        bool trackFrame(shared_ptr<FrameHessian> newFrameHessian);

        void calcTGrads(shared_ptr<FrameHessian> newFrameHessian);

        // for multi-fisheye
        void setFirstforMF(shared_ptr<FrameHessian> newFrameHessian);

        bool trackFrameforMF(shared_ptr<FrameHessian> newFrameHessian, shared_ptr<PangolinDSOViewer> viewer);

        void debugPlotforMF(int lvl, shared_ptr<PangolinDSOViewer> viewer);

        int frameID = -1;
        bool fixAffine = true;
        bool printDebug = false;

        Pnt *points[PYR_LEVELS];
        int numPoints[PYR_LEVELS];
        AffLight thisToNext_aff;
       
        SE3 thisToNext;

        // for multi-fisheye image
        vector<Pnt *> vpoints[PYR_LEVELS];
        vector<int> vnumPoints[PYR_LEVELS];
        int sumPoints[PYR_LEVELS];
        vector<vector<AffLight>> vvthisToNext_aff;  // 5 to 5


        shared_ptr<FrameHessian> firstFrame;
        shared_ptr<FrameHessian> newFrame;
    private:
        Mat33 K[PYR_LEVELS];
        Mat33 Ki[PYR_LEVELS];
        double fx[PYR_LEVELS];
        double fy[PYR_LEVELS];
        double fxi[PYR_LEVELS];
        double fyi[PYR_LEVELS];
        double cx[PYR_LEVELS];
        double cy[PYR_LEVELS];
        double cxi[PYR_LEVELS];
        double cyi[PYR_LEVELS];
        int w[PYR_LEVELS];
        int h[PYR_LEVELS];

        void makeK(shared_ptr<CalibHessian> HCalib);

        bool snapped;    // 判断位移是否足够
        int snappedAt;

        // pyramid images & levels on all levels
        Eigen::Vector3f *dINew[PYR_LEVELS];
        Eigen::Vector3f *dIFist[PYR_LEVELS];

        Eigen::DiagonalMatrix<float, 8> wM;

        // temporary buffers for H and b.
        Vec10f *JbBuffer;            // 0-7: sum(dd * dp). 8: sum(res*dd). 9: 1/(1+sum(dd*dd))=inverse hessian entry.
        Vec10f *JbBuffer_new;

        Accumulator9 acc9;
        Accumulator9 acc9SC;

        Vec3f dGrads[PYR_LEVELS];

        float alphaK;
        float alphaW;
        float regWeight;
        float couplingWeight;

        // for multi-fisheye image
        UndistortMultiFisheye* UMF;
        int camNums;
        Eigen::DiagonalMatrix<float, 56> wMF;

        // pyramid images & levels on all levels
        vector<Eigen::Vector3f *> vdINew[PYR_LEVELS];
        vector<Eigen::Vector3f *> vdIFist[PYR_LEVELS];

        vector<Vec10f *> vJbBuffer;            // 0-7: sum(dd * dp). 8: sum(res*dd). 9: 1/(1+sum(dd*dd))=inverse hessian entry.
        vector<Vec10f *> vJbBuffer_new;

        vector<Vec3f> vdGrads[PYR_LEVELS];

        Accumulator57 acc57;
        Accumulator57 acc57SC;

        int tocamnpa;


        Vec3f calcResAndGS(
                int lvl,
                Mat88f &H_out, Vec8f &b_out,
                Mat88f &H_out_sc, Vec8f &b_out_sc,
                const SE3 &refToNew, AffLight refToNew_aff,
                bool plot);


        Vec3f calcEC(int lvl); // returns OLD NERGY, NEW ENERGY, NUM TERMS.
        void optReg(int lvl);

        void propagateUp(int srcLvl);

        void propagateDown(int srcLvl);

        float rescale();

        void resetPoints(int lvl);

        void doStep(int lvl, float lambda, Vec8f inc);

        void applyStep(int lvl);

        void makeGradients(Eigen::Vector3f **data);

        void makeNN();

        // for multi-fisheyey

        Vec3f calcResAndGSforMF(
                int lvl,
                Eigen::Matrix<float, 56, 56> &H_out, Eigen::Matrix<float, 56, 1> &b_out,
                Eigen::Matrix<float, 56, 56> &H_out_sc, Eigen::Matrix<float, 56, 1> &b_out_sc,
                const SE3 &refToNew, vector<vector<AffLight>> vvrefToNew_aff,
                bool plot);

        Vec3f calcECforMF(int lvl);

        void optRegforMF(int lvl);

        void doStepforMF(int lvl, float lambda, Vec56f inc);

        void applyStepforMF(int lvl);

        void propagateDownforMF(int srcLvl);

        void propagateUpforMF(int srcLvl);

        void resetPointsforMF(int lvl);

        void makeNNforMF();

        // dxr/dXs
        inline Eigen::Matrix<float, 2, 3> computedStoR(float SphereX, float SphereY, float SphereZ, int n, int level = 0)
        {
            Eigen::Matrix2f dIL;
            dIL << 0,-1,1,0;  // x = -y , y = x

            Eigen::Matrix<float, 2, 3> duXs;

            Eigen::Matrix<float, 3, 1> RaySphere;
            RaySphere(0, 0) = SphereX;
            RaySphere(1, 0) = SphereY;
            RaySphere(2, 0) = SphereZ;

            Eigen::Matrix<float, 3, 3> RotationMat = (UMF->mvExParas[n].block<3, 3>(0, 0)).cast<float>();
            Eigen::Matrix<float, 3, 1> CamPos = UMF->mvExParas[n].block<3, 1>(0, 3).cast<float>();
            Eigen::Matrix<float, 3, 1> RayCoor = RaySphere - CamPos;
            Eigen::Matrix<float, 3, 1> RectifiedImgPtCoor = RotationMat.inverse() * RayCoor;
            float x = RectifiedImgPtCoor(0, 0);
            float y = RectifiedImgPtCoor(1, 0);
            float z = RectifiedImgPtCoor(2, 0);

            float x0r, y0r;
            float f = UMF->GetRectifiedImgFocalLength(n, level);
            x0r = UMF->mvInnerParasPR[level][n](0, 14);
            y0r = UMF->mvInnerParasPR[level][n](0, 15);

            duXs << f/z, 0, -x*f/(z*z),
                    0, f/z, -y*f/(z*z);
            
            Eigen::Matrix<float, 2, 3> dSR;
            dSR = dIL * duXs * RotationMat.inverse();

            return dSR;
        }

        // dXs/dpose  dXs/drho1
        inline void computedXs(float rho1, Mat33f R, Vec3f t, Vec3f Xc, 
                    Eigen::Matrix<float, 3, 6> * dXsdpose, Eigen::Matrix<float, 3, 1> * dXsdrho1,
                    double X, double Y, double Z)
        {

            float mSphereRadius = UMF->GetSphereRadius();
            float x,y,z;
            x = Xc(0);
            y = Xc(1);
            z = Xc(2);
            float l2 = x * x + y * y + z * z;
            float l23 = l2 * sqrt(l2);

            Eigen::Matrix<float, 3, 3> dXsdXc;
            dXsdXc << mSphereRadius  *(y * y + z * z) / l23, -x * y * mSphereRadius / l23, -x * z * mSphereRadius / l23,
                    -x * y * mSphereRadius / l23, mSphereRadius * (x * x + z * z) / l23, -y * z * mSphereRadius / l23,
                    -x * z * mSphereRadius / l23, -y * z * mSphereRadius / l23, mSphereRadius * (x * x + y * y) / l23;

            Eigen::Matrix<float, 3, 6> dXcdpose;
            dXcdpose << 1, 0, 0, 0, z, -y,
                        0, 1, 0, -z, 0, x,
                        0, 0, 1, y, -x, 0;
            
            Eigen::Matrix<float, 3, 1> dXcdroh1;

            dXcdroh1 = - R * Vec3f(X/mSphereRadius, Y/mSphereRadius, Z/mSphereRadius) / (rho1 * rho1);
            //dXcdroh1 = t;
            
            Eigen::Matrix2f dIL;
            *dXsdpose = dXsdXc * dXcdpose;
            *dXsdrho1 = dXsdXc * dXcdroh1;
            // cout << "dXsdXc :" << dXsdXc << endl;
            // cout << "dXcdroh1 :" << dXcdroh1 << endl;
        }

        Vec5f testJaccobian(int CameraNum, double RectifiedPixalx, double RectifiedPixaly,int lvl, Mat33f R, Vec3f t, float idepth, vector<vector<Eigen::Vector2f>> vvr2new_aff);
        Vec3f testJaccobian2(float X, float Y, float Z, Mat33f R, Vec3f t, float idepth);
    
        vector<float> NumericDiff(int n, double x, double y,int lvl, Mat33f R, Vec3f t, float idepth, vector<vector<Eigen::Vector2f>> vvr2new_aff);
    
    };

    /**
     * minimal flann point cloud
     */
    struct FLANNPointcloud {
        inline FLANNPointcloud() {
            num = 0;
            points = 0;
        }

        inline FLANNPointcloud(int n, Pnt *p) : num(n), points(p) {}

        int num;
        Pnt *points;

        inline size_t kdtree_get_point_count() const { return num; }

        inline float kdtree_distance(const float *p1, const size_t idx_p2, size_t /*size*/) const {
            const float d0 = p1[0] - points[idx_p2].u;
            const float d1 = p1[1] - points[idx_p2].v;
            return d0 * d0 + d1 * d1;
        }

        inline float kdtree_get_pt(const size_t idx, int dim) const {
            if (dim == 0) return points[idx].u;
            else return points[idx].v;
        }

        template<class BBOX>
        bool kdtree_get_bbox(BBOX & /* bb */) const { return false; }
    };
}

#endif // LDSO_COARSE_INITIALIZER_H_
