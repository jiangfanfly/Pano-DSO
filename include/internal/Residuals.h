#pragma once
#ifndef LDSO_RESIDUALS_H_
#define LDSO_RESIDUALS_H_

#include <memory>

using namespace std;

#include "NumTypes.h"
#include "internal/RawResidualJacobian.h"
#include "frontend/Undistort.h"

namespace ldso {

    namespace internal {

        /**
         * photometric residuals defined in DSO
         */
        class PointHessian;

        class FrameHessian;

        class CalibHessian;

        class EnergyFunctional;

        enum ResLocation {
            ACTIVE = 0, LINEARIZED, MARGINALIZED, NONE
        };

        enum ResState {
            IN = 0, OOB, OUTLIER
        }; // Residual state: inside, outside, outlier  // IN在内部, OOB 点超出图像, OUTLIER有外点 

        struct FullJacRowT {
            Eigen::Vector2f projectedTo[MAX_RES_PER_POINT];
        };

        // Photometric reprojection Error  优化目标函数
        class PointFrameResidual {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            PointFrameResidual() : J(new RawResidualJacobian) {}

            PointFrameResidual(shared_ptr<PointHessian> point_, shared_ptr<FrameHessian> host_,
                               shared_ptr<FrameHessian> target_) : J(new RawResidualJacobian) {
                point = point_;
                host = host_;
                target = target_;
                resetOOB();
            }

            /**
             * linearize the reprojection, create jacobian matrices
             * @param HCalib
             * @return the new energy
             */
            virtual double linearize(shared_ptr<CalibHessian> &HCalib);

            virtual void resetOOB() {
                state_NewEnergy = state_energy = 0;
                state_NewState = ResState::OUTLIER;
                setState(ResState::IN);
            };

            // 将state_NewState的状态更新至当前状态
            void applyRes(bool copyJacobians) {

                if (copyJacobians) {
                    if (state_state == ResState::OOB) {
                        return;
                    }

                    if (state_NewState == ResState::IN) {
                        isActiveAndIsGoodNEW = true;
                        takeData();
                    } else {
                        isActiveAndIsGoodNEW = false;
                    }
                }

                state_state = state_NewState;
                state_energy = state_NewEnergy;
            }

            void setState(ResState s) { state_state = s; }

            static int instanceCounter;
            ResState state_state = ResState::OUTLIER;   // //!< 上一次的残差状态
            double state_energy = 0;                        //!< 上一次的能量值
            ResState state_NewState = ResState::OUTLIER;    //!< 新的一次计算的状态
            double state_NewEnergy = 0;                     //!< 新的能量, 如果大于阈值则把等于阈值
            double state_NewEnergyWithOutlier = 0;          //!< 可能具有外点的能量, 可能大于阈值

            weak_ptr<PointHessian> point;
            weak_ptr<FrameHessian> host;
            weak_ptr<FrameHessian> target;
            shared_ptr<RawResidualJacobian> J = nullptr;    //!< 残差对变量的各种雅克比

            bool isNew = true;
            Eigen::Vector2f projectedTo[MAX_RES_PER_POINT]; // 从host到target的投影点 //!< 各个patch的投影坐标
            Vec3f centerProjectedTo;                        //!< patch的中心点投影 [像素x, 像素y, 新帧逆深度]

            // ==================================================================================== //
            // Energy stuffs
            inline bool isActive() const { return isActiveAndIsGoodNEW; }

            // fix the jacobians
            void fixLinearizationF(shared_ptr<EnergyFunctional> ef);

            int hostIDX = 0, targetIDX = 0;

            VecNRf res_toZeroF;             //!< 更新delta后的线性残差
            Vec8f JpJdF = Vec8f::Zero();    //!< 逆深度Jaco和位姿+光度Jaco的Hessian
            bool isLinearized = false;  // if linearization is fixed.

            // if residual is not OOB & not OUTLIER & should be used during accumulations
            bool isActiveAndIsGoodNEW = false;

            void takeData() {
                // 图像导数 * 图像导数 * 逆深度导数 // dr21/dx ^t * dr21/dx  * dx/drho
                Vec2f JI_JI_Jd = J->JIdx2 * J->Jpdd;   
                // 位姿导数 * 图像导数 * 图像导数 * 逆深度导数  // 前6 dx/dkexi  * dI/dx^t * dI/dx *dx2/drho = dr/dpose ^t * dr/drho  pose和rho hessian
                for (int i = 0; i < 6; i++)
                    JpJdF[i] = J->Jpdxi[0][i] * JI_JI_Jd[0] + J->Jpdxi[1][i] * JI_JI_Jd[1];    
                // 光度导数  * 图像导数 * 逆深度导数     // 后2 dr21/dab ^T dI/dx * dx/drho  = dr/dab * dr/drho   a,b和rho hessian
                JpJdF.segment<2>(6) = J->JabJIdx * J->Jpdd;
            }


            //==============================================
            // for multi-fisheye
            double linearizeforMF(UndistortMultiFisheye* UMF);

            // fix the jacobians
            void fixLinearizationFforMF(shared_ptr<EnergyFunctional> ef);

            void computedxrdposedrho1(UndistortMultiFisheye * UMF,const Mat33f &R, const Vec3f &t, const Vec3f &Xc, Vec3f KliP,const float &rho1 ,const int &dx, const int &dy, 
                    const float &drescale, const int n, Eigen::Matrix<float, 2, 1> &dxrdrho1, Eigen::Matrix<float, 2, 6> &dxrdpose, int level) 
            {
                Eigen::Matrix2f dIL;
                dIL << 0,-1,1,0;

                float mSphereRadius = UMF->GetSphereRadius();
                float xs = Xc(0) * drescale * mSphereRadius;
                float ys = Xc(1) * drescale * mSphereRadius;
                float zs = Xc(2) * drescale * mSphereRadius;

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

                duXs << f/zsn, 0, -xsn*f/(zsn*zsn),
                        0, f/zsn, -ysn*f/(zsn*zsn);
                
                Eigen::Matrix<float, 2, 3> dxrdXs;
                dxrdXs = dIL * duXs * RotationMat.inverse();

                float xc = Xc(0);
                float yc = Xc(1);
                float zc = Xc(2);
                float l2 = xc * xc + yc * yc + zc * zc;
                float l23 = l2 * sqrt(l2);

                Eigen::Matrix<float, 3, 3> dXsdXc;
                dXsdXc << mSphereRadius  *(yc * yc + zc * zc) / l23, -xc * yc * mSphereRadius / l23, -xc * zc * mSphereRadius / l23,
                        -xc * yc * mSphereRadius / l23, mSphereRadius * (xc * xc + zc * zc) / l23, -yc * zc * mSphereRadius / l23,
                        -xc * zc * mSphereRadius / l23, -yc * zc * mSphereRadius / l23, mSphereRadius * (xc * xc + yc * yc) / l23;
                
                Eigen::Matrix<float, 3, 6> dXcdpose;
                dXcdpose << 1, 0, 0, 0, zc, -yc,
                            0, 1, 0, -zc, 0, xc,
                            0, 0, 1, yc, -xc, 0;
                
                Eigen::Matrix<float, 3, 1> dXcdroh1;
                dXcdroh1 = - R * Vec3f(KliP(0), KliP(1), KliP(2)) / (rho1 * rho1);
                //dXcdroh1 = t * SCALE_IDEPTH;
                
                
                dxrdrho1 = dxrdXs * dXsdXc * dXcdroh1;
                dxrdpose = dxrdXs * dXsdXc * dXcdpose;

                
            }

            vector<float> NumericDiff(int n, double x, double y,int lvl, Mat33f R, Vec3f t, float idepth, Vec2f ab);
            Vec5f testJaccobian(int n, double RectifiedPixalx, double RectifiedPixaly, int lvl, Mat33f R, Vec3f t, 
                            float idepth, Vec2f ab);

            int camNums;
            int hcamnum, tcamnum;


        };
    }
}

#endif // LDSO_RESIDUALS_H_
