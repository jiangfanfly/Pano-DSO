#pragma once
#ifndef LDSO_ACCUMULATED_TOP_HESSIAN_H_
#define LDSO_ACCUMULATED_TOP_HESSIAN_H_

#include "NumTypes.h"
#include "Settings.h"

#include "internal/PointHessian.h"
#include "internal/IndexThreadReduce.h"
#include "internal/OptimizationBackend/MatrixAccumulators.h"

using namespace std;

namespace ldso {

    namespace internal {

        class EnergyFunctional;

        class AccumulatedTopHessianSSE {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            inline AccumulatedTopHessianSSE() {
                for (int tid = 0; tid < NUM_THREADS; tid++) {
                    nres[tid] = 0;
                    acc[tid] = 0;
                    nframes[tid] = 0;
                    accforMF[tid] = 0;
                }

            };

            inline ~AccumulatedTopHessianSSE() {
                for (int tid = 0; tid < NUM_THREADS; tid++) {
                    if (acc[tid] != 0) delete[] acc[tid];
                    if (accforMF[tid] != 0) delete[] accforMF[tid];
                }
            };

            inline void setZero(int nFrames, int min = 0, int max = 1, Vec10 *stats = 0, int tid = 0) {

                if (nFrames != nframes[tid]) {
                    if (acc[tid] != 0) delete[] acc[tid];
#if USE_XI_MODEL
                    acc[tid] = new Accumulator14[nFrames*nFrames];
#else
                    acc[tid] = new AccumulatorApprox[nFrames * nFrames];
#endif
                }

                for (int i = 0; i < nFrames * nFrames; i++) { acc[tid][i].initialize(); }

                nframes[tid] = nFrames;
                nres[tid] = 0;

            }

            void stitchDouble(MatXX &H, VecX &b, EnergyFunctional const *const EF, bool usePrior, bool useDelta,
                              int tid = 0);

            template<int mode>
            void addPoint(shared_ptr<PointHessian> p, EnergyFunctional const *const ef, int tid = 0);


            void stitchDoubleMT(IndexThreadReduce<Vec10>* red, MatXX &H, VecX &b, EnergyFunctional const *const EF,
                                bool usePrior, bool MT) {
                // sum up, splitting by bock in square.
                if (MT) {
                    MatXX Hs[NUM_THREADS];
                    VecX bs[NUM_THREADS];
                    for (int i = 0; i < NUM_THREADS; i++) {
                        assert(nframes[0] == nframes[i]);
                        Hs[i] = MatXX::Zero(nframes[0] * 8 + CPARS, nframes[0] * 8 + CPARS);
                        bs[i] = VecX::Zero(nframes[0] * 8 + CPARS);
                    }

                    red->reduce(bind(&AccumulatedTopHessianSSE::stitchDoubleInternal,
                                     this, Hs, bs, EF, usePrior, _1, _2, _3, _4), 0, nframes[0] * nframes[0], 0);

                    // sum up results
                    H = Hs[0];
                    b = bs[0];

                    for (int i = 1; i < NUM_THREADS; i++) {
                        H.noalias() += Hs[i];
                        b.noalias() += bs[i];
                        nres[0] += nres[i];
                    }
                } else {
                    H = MatXX::Zero(nframes[0] * 8 + CPARS, nframes[0] * 8 + CPARS);
                    b = VecX::Zero(nframes[0] * 8 + CPARS);
                    stitchDoubleInternal(&H, &b, EF, usePrior, 0, nframes[0] * nframes[0], 0, -1);
                }

                // make diagonal by copying over parts.
                for (int h = 0; h < nframes[0]; h++) {
                    int hIdx = CPARS + h * 8;
                    H.block<CPARS, 8>(0, hIdx).noalias() = H.block<8, CPARS>(hIdx, 0).transpose();

                    for (int t = h + 1; t < nframes[0]; t++) {
                        int tIdx = CPARS + t * 8;
                        H.block<8, 8>(hIdx, tIdx).noalias() += H.block<8, 8>(tIdx, hIdx).transpose();
                        H.block<8, 8>(tIdx, hIdx).noalias() = H.block<8, 8>(hIdx, tIdx).transpose();
                    }
                }
            }

            int nframes[NUM_THREADS];   ////!< 每个线程的帧数
            EIGEN_ALIGN16 AccumulatorApprox *acc[NUM_THREADS];      //!< 计算hessian的累乘器
            int nres[NUM_THREADS];          //!< 残差计数

            template<int mode>
            inline void addPointsInternal(
                    std::vector<shared_ptr<PointHessian>> *points, EnergyFunctional const *const ef,
                    int min = 0, int max = 1, Vec10 *stats = 0, int tid = 0) {
                for (int i = min; i < max; i++)
                    addPoint<mode>((*points)[i], ef, tid);
            }

            //========================================
            // for-multi-fisheye
            inline void setZeroforMF(int nFrames, int min = 0, int max = 1, Vec10 *stats = 0, int tid = 0) {

                if (nFrames != nframes[tid]) {
                    if (accforMF[tid] != 0) delete[] accforMF[tid];
#if USE_XI_MODEL
                    accforMF[tid] = new AccumulatorNoschur[nFrames*nFrames];
#else
                    accforMF[tid] = new AccumulatorNoschur[nFrames * nFrames];
#endif
                }

                for (int i = 0; i < nFrames * nFrames; i++) { accforMF[tid][i].initialize(); }

                nframes[tid] = nFrames;
                nres[tid] = 0;

            }

            void stitchDoubleforMF(MatXX &H, VecX &b, EnergyFunctional const *const EF, bool usePrior, bool useDelta,
                              int tid = 0);

            template<int mode>
            void addPointforMF(shared_ptr<PointHessian> p, EnergyFunctional const *const ef, int tid = 0);

            template<int mode>
            inline void addPointsInternalforMF(
                    std::vector<shared_ptr<PointHessian>> *points, EnergyFunctional const *const ef,
                    int min = 0, int max = 1, Vec10 *stats = 0, int tid = 0) {
                for (int i = min; i < max; i++)
                    addPointforMF<mode>((*points)[i], ef, tid);
            }

            void stitchDoubleMTforMF(IndexThreadReduce<Vec10>* red, MatXX &H, VecX &b, EnergyFunctional const *const EF,
                                bool usePrior, bool MT) 
            {
                // sum up, splitting by bock in square.
                if (MT) 
                {
                    MatXX Hs[NUM_THREADS];
                    VecX bs[NUM_THREADS];
                    for (int i = 0; i < NUM_THREADS; i++) 
                    {
                        assert(nframes[0] == nframes[i]);
                        //* 所有的优化变量维度 num frames*(6 pose + 5*2 光度仿射 )
                        Hs[i] = MatXX::Zero(nframes[0] * 16, nframes[0] * 16);
                        bs[i] = VecX::Zero(nframes[0] * 16);
                    }

                    red->reduce(bind(&AccumulatedTopHessianSSE::stitchDoubleInternalforMF,
                                     this, Hs, bs, EF, usePrior, _1, _2, _3, _4), 0, nframes[0] * nframes[0], 0);

                    // sum up results
                    H = Hs[0];
                    b = bs[0];

                    for (int i = 1; i < NUM_THREADS; i++) 
                    {
                        H.noalias() += Hs[i];
                        b.noalias() += bs[i];
                        nres[0] += nres[i];
                    }
                } 
                else 
                {
                    H = MatXX::Zero(nframes[0] * 16 , nframes[0] * 16);
                    b = VecX::Zero(nframes[0] * 16);
                    stitchDoubleInternalforMF(&H, &b, EF, usePrior, 0, nframes[0] * nframes[0], 0, -1);
                }

                // make diagonal by copying over parts.
                for (int h = 0; h < nframes[0]; h++) 
                {
                    int hIdx =h * 16;
                    for (int t = h + 1; t < nframes[0]; t++) 
                    {
                        int tIdx = t * 16;
                        //! 对于位姿, 相同两帧之间的Hessian需要加起来, 即对称位置的, (J差负号, 平方之后就好了)
                        H.block<16, 16>(hIdx, tIdx).noalias() += H.block<16, 16>(tIdx, hIdx).transpose();
                        H.block<16, 16>(tIdx, hIdx).noalias() = H.block<16, 16>(hIdx, tIdx).transpose();
                    }
                }
            }

            
            EIGEN_ALIGN16 AccumulatorNoschur *accforMF[NUM_THREADS];   // 8*8个 i帧与j帧    //!< 计算hessian的累乘器



        private:

            void stitchDoubleInternal(
                    MatXX *H, VecX *b, EnergyFunctional const *const EF, bool usePrior,
                    int min, int max, Vec10 *stats, int tid);

            // for multi-fisheye
            void stitchDoubleInternalforMF(
                    MatXX *H, VecX *b, EnergyFunctional const *const EF, bool usePrior,
                    int min, int max, Vec10 *stats, int tid);
        };
    }
}

#endif