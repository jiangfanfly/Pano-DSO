#pragma once
#ifndef LDSO_FRAME_HESSIAN_H_
#define LDSO_FRAME_HESSIAN_H_

#include "Frame.h"
#include "NumTypes.h"
#include "Settings.h"
#include "AffLight.h"

#include "internal/FrameFramePrecalc.h"
#include "frontend/Undistort.h"

using namespace std;

namespace ldso {

    namespace internal {

        class PointHessian;

        class CalibHessian;

        struct FrameFramePrecalc;

        /**
         * Frame hessian is the internal structure used in dso
         */
        class FrameHessian {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            FrameHessian(shared_ptr<Frame> frame) {
                this->frame = frame;
            }

            FrameHessian(shared_ptr<Frame> frame, UndistortMultiFisheye* u) 
            {
                this->frame = frame;
                this->UMF = u;
                camNums = UMF->camNums;
            }

            ~FrameHessian() 
            {
                // for (int i = 0; i < pyrLevelsUsed; i++) {
                //     delete[] dIp[i];
                //     delete[]  absSquaredGrad[i];
                // }
                
                for (int n = 0; n < camNums; n++)
                {
                    for (int i = 0; i < pyrLevelsUsed; i++)
                    {
                        delete[] vdIp[i][n];
                        delete[] vabsSquaredGrad[i][n];
                        delete[] vmask[i][n];
                    }
                    delete[] vdfisheyeI[n];
                }
                
            }

            // accessors
            EIGEN_STRONG_INLINE const SE3 &get_worldToCam_evalPT() const {
                return worldToCam_evalPT;
            }

            EIGEN_STRONG_INLINE const Vec10 &get_state_zero() const {
                return state_zero;
            }

            EIGEN_STRONG_INLINE const Vec10 &get_state() const {
                return state;
            }

            EIGEN_STRONG_INLINE const Vec10 &get_state_scaled() const {
                return state_scaled;
            }

            // state - state0
            EIGEN_STRONG_INLINE const Vec10 get_state_minus_stateZero() const {
                return get_state() - get_state_zero();
            }

            inline Vec6 w2c_leftEps() const {
                return get_state_scaled().head<6>();
            }

            inline AffLight aff_g2l() {
                return AffLight(get_state_scaled()[6], get_state_scaled()[7]);
            }

            inline AffLight aff_g2l_0() const {
                return AffLight(get_state_zero()[6] * SCALE_A, get_state_zero()[7] * SCALE_B);
            }

            void setStateZero(const Vec10 &state_zero);

            inline void setState(const Vec10 &state) {

                this->state = state;
                state_scaled.segment<3>(0) = SCALE_XI_TRANS * state.segment<3>(0);
                state_scaled.segment<3>(3) = SCALE_XI_ROT * state.segment<3>(3);
                state_scaled[6] = SCALE_A * state[6];
                state_scaled[7] = SCALE_B * state[7];
                state_scaled[8] = SCALE_A * state[8];
                state_scaled[9] = SCALE_B * state[9];

                PRE_worldToCam = SE3::exp(w2c_leftEps()) * get_worldToCam_evalPT();
                PRE_camToWorld = PRE_worldToCam.inverse();
            };

            inline void setStateScaled(const Vec10 &state_scaled) {

                this->state_scaled = state_scaled;
                state.segment<3>(0) = SCALE_XI_TRANS_INVERSE * state_scaled.segment<3>(0);
                state.segment<3>(3) = SCALE_XI_ROT_INVERSE * state_scaled.segment<3>(3);
                state[6] = SCALE_A_INVERSE * state_scaled[6];
                state[7] = SCALE_B_INVERSE * state_scaled[7];
                state[8] = SCALE_A_INVERSE * state_scaled[8];
                state[9] = SCALE_B_INVERSE * state_scaled[9];

                PRE_worldToCam = SE3::exp(w2c_leftEps()) * get_worldToCam_evalPT();
                PRE_camToWorld = PRE_worldToCam.inverse();
            };

            inline void setEvalPT(const SE3 &worldToCam_evalPT, const Vec10 &state) {

                this->worldToCam_evalPT = worldToCam_evalPT;
                setState(state);
                setStateZero(state);
            };

            // set the pose Tcw
            inline void  setEvalPT_scaled(const SE3 &worldToCam_evalPT, const AffLight &aff_g2l) {
                Vec10 initial_state = Vec10::Zero();
                initial_state[6] = aff_g2l.a;
                initial_state[7] = aff_g2l.b;
                this->worldToCam_evalPT = worldToCam_evalPT;
                setStateScaled(initial_state);
                setStateZero(this->get_state());
            };

            /**
             * @brief create the images and gradient from original image
             * @param [in] HCalib camera intrinsics with hessian
             */
            void makeImages(float *image, const shared_ptr<CalibHessian> &HCalib);

            inline Vec10 getPrior() {
                Vec10 p = Vec10::Zero();
                if (frame->id == 0) {
                    p.head<3>() = Vec3::Constant(setting_initialTransPrior);
                    p.segment<3>(3) = Vec3::Constant(setting_initialRotPrior);
                    if (setting_solverMode & SOLVER_REMOVE_POSEPRIOR) {
                        p.head<6>().setZero();
                    }
                    p[6] = setting_initialAffAPrior;
                    p[7] = setting_initialAffBPrior;
                } else {
                    if (setting_affineOptModeA < 0) {
                        p[6] = setting_initialAffAPrior;
                    } else {
                        p[6] = setting_affineOptModeA;
                    }
                    if (setting_affineOptModeB < 0) {
                        p[7] = setting_initialAffBPrior;
                    } else {
                        p[7] = setting_affineOptModeB;
                    }
                }
                p[8] = setting_initialAffAPrior;
                p[9] = setting_initialAffBPrior;
                return p;
            }

            inline Vec10 getPriorZero() {
                return Vec10::Zero();
            }

            //====================================
            // for multi-fisheye
            // create multi fisheye images
            void makeImages(vector<float *> vimage, UndistortMultiFisheye* UMF);

            inline void setEvalPTforMF(const SE3 &worldToCam_evalPT, const Vec16 &state) 
            {
                this->worldToCam_evalPT = worldToCam_evalPT;
                setStateforMF(state);
                setStateZeroforMF(state);
            };

            // set the pose Tcw
            inline void  setEvalPT_scaledforMF(const SE3 &worldToCam_evalPT, const vector<AffLight> &vaff_g2l) 
            {
                Vec16 initial_stateforMF = Vec16::Zero();
                for(int n=0; n < vaff_g2l.size(); n++)
                {
                    initial_stateforMF[6+2*n] = vaff_g2l[n].a;
                    initial_stateforMF[7+2*n] = vaff_g2l[n].b;
                }

                this->worldToCam_evalPT = worldToCam_evalPT;
                setStateScaledforMF(initial_stateforMF);
                setStateZeroforMF(this->get_stateforMF());
            };

            inline void setStateforMF(const Vec16 &stateforMF) {

                this->stateforMF = stateforMF;
                state_scaledforMF.segment<3>(0) = SCALE_XI_TRANS * stateforMF.segment<3>(0);
                state_scaledforMF.segment<3>(3) = SCALE_XI_ROT * stateforMF.segment<3>(3);
                for(int n = 0; n < camNums; n++)
                {
                    state_scaledforMF[6 + 2*n] = SCALE_A * stateforMF[6 + 2*n];
                    state_scaledforMF[7 + 2*n] = SCALE_B * stateforMF[7 + 2*n];
                }

                PRE_worldToCam = SE3::exp(w2c_leftEps()) * get_worldToCam_evalPT();
                PRE_camToWorld = PRE_worldToCam.inverse();
            };

            inline void setStateScaledforMF(const Vec16 &state_scaledforMF) {

                this->state_scaledforMF = state_scaledforMF;
                stateforMF.segment<3>(0) = SCALE_XI_TRANS_INVERSE * state_scaledforMF.segment<3>(0);
                stateforMF.segment<3>(3) = SCALE_XI_ROT_INVERSE * state_scaledforMF.segment<3>(3);
                for(int n = 0; n < camNums; n++)
                {
                    stateforMF[6 + 2*n] = SCALE_A_INVERSE * state_scaledforMF[6 + 2*n];
                    stateforMF[7 + 2*n] = SCALE_B_INVERSE * state_scaledforMF[7 + 2*n];
                }
                PRE_worldToCam = SE3::exp(w2c_leftEps()) * get_worldToCam_evalPT();
                PRE_camToWorld = PRE_worldToCam.inverse();
            };

            // EIGEN_STRONG_INLINE const SE3 &get_worldToCam_evalPTforMF() const {
            //     return worldToCam_evalPT;
            // }

            EIGEN_STRONG_INLINE const Vec16 &get_state_zeroforMF() const 
            {
                return state_zeroforMF;
            }

            EIGEN_STRONG_INLINE const Vec16 &get_stateforMF() const 
            {
                return stateforMF;
            }

            EIGEN_STRONG_INLINE const Vec16 &get_state_scaledforMF() const 
            {
                return state_scaledforMF;
            }

            // state - state0
            EIGEN_STRONG_INLINE const Vec16 get_state_minus_stateZeroforMF() const 
            {
                // Vec26 gs = get_stateforMF();
                // Vec26 gsz = get_state_zeroforMF();
                // Vec56 gssz;
                // gssz.head<6>() = gs.head<6>() - gsz.head<6>();
                // for(int n1 = 0; n1 < camNums; n1++)
                // {
                //     for(int n2 = 0; n2 < camNums; n2++)
                //     {
                //         int s = n1 * camNums + n2;
                //         gssz[6 + s*2] = gs[6 + 2 * n1] - gsz[6 + 2 * n2];
                //         gssz[7 + s*2] = gs[7 + 2 * n1] - gsz[7 + 2 * n2];
                //     }
                // }
                // return gssz;

                Vec16 gs = get_stateforMF();
                Vec16 gpsz = get_state_zeroforMF();

                return (gs-gpsz);
            }

            EIGEN_STRONG_INLINE const Vec56 get_state_minus_stateZeroforMF2() const 
            {
                Vec16 gs = get_stateforMF();
                Vec16 gsz = get_state_zeroforMF();
                Vec56 gssz;
                gssz.head<6>() = gs.head<6>() - gsz.head<6>();
                for(int n1 = 0; n1 < camNums; n1++)
                {
                    for(int n2 = 0; n2 < camNums; n2++)
                    {
                        int s = n1 * camNums + n2;
                        gssz[6 + s*2] = gs[6 + 2 * n1] - gsz[6 + 2 * n2];
                        gssz[7 + s*2] = gs[7 + 2 * n1] - gsz[7 + 2 * n2];
                    }
                }
                return gssz;
            }

            EIGEN_STRONG_INLINE const Vec16 get_state_minus_PriorstateZeroforMF() const 
            {
                // Vec26 gs = get_stateforMF();
                // Vec26 gpsz = getPriorZeroforMF();
                // Vec56 gpssz;
                // gpssz.head<6>() = gs.head<6>() - gpsz.head<6>();
                // for(int n1 = 0; n1 < camNums; n1++)
                // {
                //     for(int n2 = 0; n2 < camNums; n2++)
                //     {
                //         int s = n1 * camNums + n2;
                //         gpssz[6 + s*2] = gs[6 + 2 * n1] - gpsz[6 + 2 * n2];
                //         gpssz[7 + s*2] = gs[7 + 2 * n1] - gpsz[7 + 2 * n2];
                //     }
                // }
                // return gpssz;

                Vec16 gs = get_stateforMF();
                Vec16 gpsz = getPriorZeroforMF();

                return (gs-gpsz);
            }

            inline Vec16 getPriorforMF() 
            {
                Vec16 p = Vec16::Zero();
                if (frame->id == 0) 
                {
                    p.head<3>() = Vec3::Constant(setting_initialTransPrior);
                    p.segment<3>(3) = Vec3::Constant(setting_initialRotPrior);
                    if (setting_solverMode & SOLVER_REMOVE_POSEPRIOR) 
                    {
                        p.head<6>().setZero();
                    }
                    for(int n = 0; n < camNums; n++)
                    {
                        p[6 + 2*n] = setting_initialAffAPrior;
                        p[7 + 2*n] = setting_initialAffBPrior;
                    }    
                } 
                else 
                {
                    if (setting_affineOptModeA < 0) {
                        p[6] = setting_initialAffAPrior;
                        p[8] = setting_initialAffAPrior;
                        p[10] = setting_initialAffAPrior;
                        p[12] = setting_initialAffAPrior;
                        p[14] = setting_initialAffAPrior;
                    } else {
                        p[6] = setting_affineOptModeA;
                        p[8] = setting_affineOptModeA;
                        p[10] = setting_affineOptModeA;
                        p[12] = setting_affineOptModeA;
                        p[14] = setting_affineOptModeA;
                    }
                    if (setting_affineOptModeB < 0) {
                        p[7] = setting_initialAffBPrior;
                        p[9] = setting_initialAffBPrior;
                        p[11] = setting_initialAffBPrior;
                        p[13] = setting_initialAffBPrior;
                        p[15] = setting_initialAffBPrior;
                    } else {
                        p[7] = setting_affineOptModeB;
                        p[9] = setting_affineOptModeB;
                        p[11] = setting_affineOptModeB;
                        p[13] = setting_affineOptModeB;
                        p[15] = setting_affineOptModeB;
                    }
                }
                // p[8] = setting_initialAffAPrior;
                // p[9] = setting_initialAffBPrior;
                return p;
            }

            inline Vec16 getPriorZeroforMF() const {
                return Vec16::Zero();
            }

            inline Vec6 w2c_leftEpsforMF() const 
            {
                return get_state_scaledforMF().head<6>();
            }

            inline vector<AffLight> aff_g2lforMF() 
            {
                vector<AffLight> va;
                for(int n = 0; n < camNums; n++)
                {
                    va.emplace_back(AffLight(get_state_scaledforMF()[6 + 2*n], get_state_scaledforMF()[7 + 2*n]));
                }
                return va;
            }

            inline vector<AffLight> aff_g2l_0forMF() const 
            {
                vector<AffLight> va;
                for(int n = 0; n < camNums; n++)
                {
                    va.emplace_back(AffLight(get_state_zeroforMF()[6 + 2*n] * SCALE_A, get_state_zeroforMF()[7 + 2*n] * SCALE_B));
                }
                return va;
            }

            void setStateZeroforMF(const Vec16 &state_zero);


            // Data
            int frameID = 0;              // key-frame ID, will be set when adding new keyframes
            shared_ptr<Frame> frame = nullptr;    // link to original frame

            // internal structures used in DSO
            // image pyramid and gradient image
            // dIp[i] is the i-th pyramid with dIp[i][0] is the original image，[1] is dx and [2] is dy
            // by default, we have 6 pyramids, so we have dIp[0...5]
            // created in makeImages()
            Vec3f *dIp[PYR_LEVELS];
            // absolute squared gradient of each pyramid
            float *absSquaredGrad[PYR_LEVELS];  // only used for pixel select (histograms etc.). no NAN.
            // dI = dIp[0], the first pyramid
            Vec3f *dI = nullptr;     // trace, fine tracking. Used for direction select (not for gradient histograms etc.)
            
            // Photometric Calibration Stuff
            float frameEnergyTH = 8 * 8 * patternNum;    // set dynamically depending on tracking residual
            float ab_exposure = 0;  // the exposure time // 曝光时间

            bool flaggedForMarginalization = false; // flag for margin
            Mat66 nullspaces_pose = Mat66::Zero();
            Mat42 nullspaces_affine = Mat42::Zero();
            Vec6 nullspaces_scale = Vec6::Zero();

            // variable info.
            SE3 worldToCam_evalPT;  // Tcw (in ORB-SLAM's framework)

            // state variable，[0-5] is se3, 6-7 is light param a,b
            Vec10 state;        // [0-5: worldToCam-leftEps. 6-7: a,b]    状态变量未经光度参数a,b矫正

            // variables used in optimization
            Vec10 step = Vec10::Zero();
            Vec10 step_backup = Vec10::Zero();
            Vec10 state_backup = Vec10::Zero();
            Vec10 state_zero = Vec10::Zero();
            Vec10 state_scaled = Vec10::Zero();    // 状态变量经过光度参数a,b矫正后

            // precalculated values, will be send to frame when optimization is done.
            SE3 PRE_worldToCam; // TCW
            SE3 PRE_camToWorld; // TWC
            // 对于其它帧的预运算值
            std::vector<FrameFramePrecalc, Eigen::aligned_allocator<FrameFramePrecalc>> targetPrecalc;

            // ======================================================================================== //
            // Energy stuffs
            // Frame status: 6 dof pose + 2 dof light param
            void takeData();        // take data from frame hessian
            Vec8 prior = Vec8::Zero();             // prior hessian (diagonal)
            Vec8 delta_prior = Vec8::Zero();       // = state-state_prior (E_prior = (delta_prior)' * diag(prior) * (delta_prior)
            Vec8 delta = Vec8::Zero();             // state - state_zero.
            int idx = 0;                         // the id in the sliding window, used for constructing matricies


            //================================== for multi-fisheye
            // multi-fisheye images
            vector<Vec3f *> vdIp[PYR_LEVELS];  // 纠正后图像梯度   0 存纠正后 1 (纠正后)x方向梯度， 2 (纠正后)y方向梯度，
            // multi-fisheye absolute squared gradient of each pyramid
            vector<float *> vabsSquaredGrad[PYR_LEVELS];
            vector<Vec3f *> vdI;            // 第一层lvl 纠正后图像及梯度
            vector<float *> vdfisheyeI;     // 原始鱼眼
            vector<float *> vmask[PYR_LEVELS];    // 纠正后影像黑边 mask掩膜
            // number of fisheye image 
            int camNums;
            UndistortMultiFisheye* UMF;

            // state variable，[0-5] is se3, 6-7 is light param a,b
            Vec16 stateforMF;        // [0-5: worldToCam-leftEps. 6-7: a,b]    状态变量未经光度参数a,b矫正后

            // variables used in optimization
            Vec16 stepforMF = Vec16::Zero();
            Vec16 step_backupforMF = Vec16::Zero();
            Vec16 state_backupforMF = Vec16::Zero();
            Vec16 state_zeroforMF = Vec16::Zero();
            Vec16 state_scaledforMF = Vec16::Zero();    // 状态变量经过光度参数a,b矫正后

            // Energy stuffs
            // Frame status: 6 dof pose + 2 dof light param * camNum * camNUM
            void takeDataforMF();        // take data from frame hessian
            Vec16 priorforMF = Vec16::Zero();             // prior hessian (diagonal)
            Vec16 delta_priorforMF = Vec16::Zero();       // = state-state_prior (E_prior = (delta_prior)' * diag(prior) * (delta_prior)
            Vec16 deltaforMF = Vec16::Zero();             // state - state_zero. 

            // Mat42 nullspaces_affine = Mat42::Zero();                    

        };
    }
}

#endif // LDSO_FRAME_HESSIAN_H_
