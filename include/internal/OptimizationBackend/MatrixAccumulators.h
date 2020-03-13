#pragma once
#ifndef LDSO_MATRIX_ACCUMULATORS_H_
#define LDSO_MATRIX_ACCUMULATORS_H_

#include "NumTypes.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace ldso {
    namespace internal {

        /**
         * Matrix accumulators with different sizes
         * SSE accelerated in some partial specializations
         * @tparam i
         * @tparam j
         */
        template<int i, int j>
        class AccumulatorXX {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            Eigen::Matrix<float, i, j> A;
            Eigen::Matrix<float, i, j> A1k;
            Eigen::Matrix<float, i, j> A1m;
            size_t num;

            inline void initialize() {
                A.setZero();
                A1k.setZero();
                A1m.setZero();
                num = numIn1 = numIn1k = numIn1m = 0;
            }

            inline void finish() {
                shiftUp(true);
                num = numIn1 + numIn1k + numIn1m;
            }


            inline void update(const Eigen::Matrix<float, i, 1> &L, const Eigen::Matrix<float, j, 1> &R, float w) {
                A += w * L * R.transpose();
                numIn1++;
                shiftUp(false);
            }


        private:
            float numIn1, numIn1k, numIn1m;

            void shiftUp(bool force) {
                if (numIn1 > 1000 || force) {
                    A1k += A;
                    A.setZero();
                    numIn1k += numIn1;
                    numIn1 = 0;
                }
                if (numIn1k > 1000 || force) {
                    A1m += A1k;
                    A1k.setZero();
                    numIn1m += numIn1k;
                    numIn1k = 0;
                }
            }
        };

        // for multi-fisheye
        class Accumulator5656 {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            Mat5656f A;
            Mat5656f A1k;
            Mat5656f A1m;
            size_t num;

            inline void initialize() {
                A.setZero();
                A1k.setZero();
                A1m.setZero();
                num = numIn1 = numIn1k = numIn1m = 0;
            }

            inline void finish() {
                shiftUp(true);
                num = numIn1 + numIn1k + numIn1m;
            }


            inline void update(const Vec8f &L, const Vec8f &R, float w, int lhc, int ltc, int rhc, int rtc) 
            {
                Vec56f r, l;
                r.setZero();
                l.setZero();
                l.segment<6>(0) = L.segment<6>(0);
                r.segment<6>(0) = R.segment<6>(0);
                int lidx = 6 + (lhc * 5 + ltc) * 2;
                int ridx = 6 + (rhc * 5 + rtc) * 2;
                l.segment<2>(lidx) = L.segment<2>(6);
                r.segment<2>(ridx) = R.segment<2>(6);
                A += w * l * r.transpose();
                numIn1++;
                shiftUp(false);
            }


        private:
            float numIn1, numIn1k, numIn1m;

            void shiftUp(bool force) {
                if (numIn1 > 1000 || force) {
                    A1k += A;
                    A.setZero();
                    numIn1k += numIn1;
                    numIn1 = 0;
                }
                if (numIn1k > 1000 || force) {
                    A1m += A1k;
                    A1k.setZero();
                    numIn1m += numIn1k;
                    numIn1k = 0;
                }
            }
        };

        class Accumulator11 {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            float A;
            size_t num;

            inline void initialize() {
                A = 0;
                memset(SSEData, 0, sizeof(float) * 4 * 1);
                memset(SSEData1k, 0, sizeof(float) * 4 * 1);
                memset(SSEData1m, 0, sizeof(float) * 4 * 1);
                num = numIn1 = numIn1k = numIn1m = 0;
            }

            inline void finish() {
                shiftUp(true);
                A = SSEData1m[0 + 0] + SSEData1m[0 + 1] + SSEData1m[0 + 2] + SSEData1m[0 + 3];
            }


            inline void updateSingle(
                    const float val) {
                SSEData[0] += val;
                num++;
                numIn1++;
                shiftUp(false);
            }

            inline void updateSSE(
                    const __m128 val) {
                _mm_store_ps(SSEData, _mm_add_ps(_mm_load_ps(SSEData), val));
                num += 4;
                numIn1++;
                shiftUp(false);
            }

            inline void updateSingleNoShift(
                    const float val) {
                SSEData[0] += val;
                num++;
                numIn1++;
            }

            inline void updateSSENoShift(
                    const __m128 val) {
                _mm_store_ps(SSEData, _mm_add_ps(_mm_load_ps(SSEData), val));
                num += 4;
                numIn1++;
            }


        private:
            EIGEN_ALIGN16 float SSEData[4 * 1];
            EIGEN_ALIGN16 float SSEData1k[4 * 1];
            EIGEN_ALIGN16 float SSEData1m[4 * 1];
            float numIn1, numIn1k, numIn1m;


            void shiftUp(bool force) {
                if (numIn1 > 1000 || force) {
                    _mm_store_ps(SSEData1k, _mm_add_ps(_mm_load_ps(SSEData), _mm_load_ps(SSEData1k)));
                    numIn1k += numIn1;
                    numIn1 = 0;
                    memset(SSEData, 0, sizeof(float) * 4 * 1);
                }

                if (numIn1k > 1000 || force) {
                    _mm_store_ps(SSEData1m, _mm_add_ps(_mm_load_ps(SSEData1k), _mm_load_ps(SSEData1m)));
                    numIn1m += numIn1k;
                    numIn1k = 0;
                    memset(SSEData1k, 0, sizeof(float) * 4 * 1);
                }
            }
        };


        template<int i>
        class AccumulatorX {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            Eigen::Matrix<float, i, 1> A;
            Eigen::Matrix<float, i, 1> A1k;
            Eigen::Matrix<float, i, 1> A1m;
            size_t num;

            inline void initialize() {
                A.setZero();
                A1k.setZero();
                A1m.setZero();
                num = numIn1 = numIn1k = numIn1m = 0;
            }

            inline void finish() {
                shiftUp(true);
                num = numIn1 + numIn1k + numIn1m;
            }


            inline void update(const Eigen::Matrix<float, i, 1> &L, float w) {
                A += w * L;
                numIn1++;
                shiftUp(false);
            }

            inline void updateNoWeight(const Eigen::Matrix<float, i, 1> &L) {
                A += L;
                numIn1++;
                shiftUp(false);
            }

        private:
            float numIn1, numIn1k, numIn1m;

            void shiftUp(bool force) {
                if (numIn1 > 1000 || force) {
                    A1k += A;
                    A.setZero();
                    numIn1k += numIn1;
                    numIn1 = 0;
                }
                if (numIn1k > 1000 || force) {
                    A1m += A1k;
                    A1k.setZero();
                    numIn1m += numIn1k;
                    numIn1k = 0;
                }
            }
        };

        class Accumulator56 {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            Vec56f A;
            Vec56f A1k;
            Vec56f A1m;
            size_t num;

            inline void initialize() {
                A.setZero();
                A1k.setZero();
                A1m.setZero();
                num = numIn1 = numIn1k = numIn1m = 0;
            }

            inline void finish() {
                shiftUp(true);
                num = numIn1 + numIn1k + numIn1m;
            }


            inline void update(const Eigen::Matrix<float, 8, 1> &L, float w, int lhc, int ltc) 
            {
                Vec56f l;
                l.setZero();
                l.segment<6>(0) = L.segment<6>(0);
                int lidx = 6 + (lhc * 5 + ltc) * 2;
                l.segment<2>(lidx) = L.segment<2>(6);
                A += w * l;
                numIn1++;
                shiftUp(false);
            }

            inline void updateNoWeight(const Eigen::Matrix<float, 8, 1> &L, int lhc, int ltc) 
            {
                Vec56f l;
                l.setZero();
                l.segment<6>(0) = L.segment<6>(0);
                int lidx = 6 + (lhc * 5 + ltc) * 2;
                l.segment<2>(lidx) = L.segment<2>(6);
                A += l;
                numIn1++;
                shiftUp(false);
            }

        private:
            float numIn1, numIn1k, numIn1m;

            void shiftUp(bool force) {
                if (numIn1 > 1000 || force) {
                    A1k += A;
                    A.setZero();
                    numIn1k += numIn1;
                    numIn1 = 0;
                }
                if (numIn1k > 1000 || force) {
                    A1m += A1k;
                    A1k.setZero();
                    numIn1m += numIn1k;
                    numIn1k = 0;
                }
            }
        };


        class Accumulator14 {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            Mat1414f H;
            Vec14f b;
            size_t num;

            inline void initialize() {
                H.setZero();
                b.setZero();
                memset(SSEData, 0, sizeof(float) * 4 * 105);
                memset(SSEData1k, 0, sizeof(float) * 4 * 105);
                memset(SSEData1m, 0, sizeof(float) * 4 * 105);
                num = numIn1 = numIn1k = numIn1m = 0;
            }

            inline void finish() {
                H.setZero();
                shiftUp(true);
                assert(numIn1 == 0);
                assert(numIn1k == 0);

                int idx = 0;
                for (int r = 0; r < 14; r++)
                    for (int c = r; c < 14; c++) {
                        float d = SSEData1m[idx + 0] + SSEData1m[idx + 1] + SSEData1m[idx + 2] + SSEData1m[idx + 3];
                        H(r, c) = H(c, r) = d;
                        idx += 4;
                    }
                assert(idx == 4 * 105);
                num = numIn1 + numIn1k + numIn1m;
            }


            inline void updateSSE(
                    const __m128 J0, const __m128 J1,
                    const __m128 J2, const __m128 J3,
                    const __m128 J4, const __m128 J5,
                    const __m128 J6, const __m128 J7,
                    const __m128 J8, const __m128 J9,
                    const __m128 J10, const __m128 J11,
                    const __m128 J12, const __m128 J13) {
                float *pt = SSEData;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J0)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J1)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J2)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J3)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J4)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J5)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J6)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J7)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J8)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J9)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J10)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J11)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J12)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J13)));
                pt += 4;

                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J1)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J2)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J3)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J4)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J5)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J6)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J7)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J8)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J9)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J10)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J11)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J12)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J13)));
                pt += 4;

                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J2)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J3)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J4)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J5)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J6)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J7)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J8)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J9)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J10)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J11)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J12)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J13)));
                pt += 4;

                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J3)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J4)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J5)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J6)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J7)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J8)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J9)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J10)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J11)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J12)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J13)));
                pt += 4;

                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J4)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J5)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J6)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J7)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J8)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J9)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J10)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J11)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J12)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J13)));
                pt += 4;

                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J5)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J6)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J7)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J8)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J9)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J10)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J11)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J12)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J13)));
                pt += 4;

                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J6)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J7)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J8)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J9)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J10)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J11)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J12)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J13)));
                pt += 4;

                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7, J7)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7, J8)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7, J9)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7, J10)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7, J11)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7, J12)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7, J13)));
                pt += 4;

                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J8, J8)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J8, J9)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J8, J10)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J8, J11)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J8, J12)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J8, J13)));
                pt += 4;

                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J9, J9)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J9, J10)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J9, J11)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J9, J12)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J9, J13)));
                pt += 4;

                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J10, J10)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J10, J11)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J10, J12)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J10, J13)));
                pt += 4;

                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J11, J11)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J11, J12)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J11, J13)));
                pt += 4;

                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J12, J12)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J12, J13)));
                pt += 4;

                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J13, J13)));
                pt += 4;

                num += 4;
                numIn1++;
                shiftUp(false);
            }


            inline void updateSingle(
                    const float J0, const float J1,
                    const float J2, const float J3,
                    const float J4, const float J5,
                    const float J6, const float J7,
                    const float J8, const float J9,
                    const float J10, const float J11,
                    const float J12, const float J13,
                    int off = 0) {
                float *pt = SSEData + off;
                *pt += J0 * J0;
                pt += 4;
                *pt += J1 * J0;
                pt += 4;
                *pt += J2 * J0;
                pt += 4;
                *pt += J3 * J0;
                pt += 4;
                *pt += J4 * J0;
                pt += 4;
                *pt += J5 * J0;
                pt += 4;
                *pt += J6 * J0;
                pt += 4;
                *pt += J7 * J0;
                pt += 4;
                *pt += J8 * J0;
                pt += 4;
                *pt += J9 * J0;
                pt += 4;
                *pt += J10 * J0;
                pt += 4;
                *pt += J11 * J0;
                pt += 4;
                *pt += J12 * J0;
                pt += 4;
                *pt += J13 * J0;
                pt += 4;

                *pt += J1 * J1;
                pt += 4;
                *pt += J2 * J1;
                pt += 4;
                *pt += J3 * J1;
                pt += 4;
                *pt += J4 * J1;
                pt += 4;
                *pt += J5 * J1;
                pt += 4;
                *pt += J6 * J1;
                pt += 4;
                *pt += J7 * J1;
                pt += 4;
                *pt += J8 * J1;
                pt += 4;
                *pt += J9 * J1;
                pt += 4;
                *pt += J10 * J1;
                pt += 4;
                *pt += J11 * J1;
                pt += 4;
                *pt += J12 * J1;
                pt += 4;
                *pt += J13 * J1;
                pt += 4;

                *pt += J2 * J2;
                pt += 4;
                *pt += J3 * J2;
                pt += 4;
                *pt += J4 * J2;
                pt += 4;
                *pt += J5 * J2;
                pt += 4;
                *pt += J6 * J2;
                pt += 4;
                *pt += J7 * J2;
                pt += 4;
                *pt += J8 * J2;
                pt += 4;
                *pt += J9 * J2;
                pt += 4;
                *pt += J10 * J2;
                pt += 4;
                *pt += J11 * J2;
                pt += 4;
                *pt += J12 * J2;
                pt += 4;
                *pt += J13 * J2;
                pt += 4;

                *pt += J3 * J3;
                pt += 4;
                *pt += J4 * J3;
                pt += 4;
                *pt += J5 * J3;
                pt += 4;
                *pt += J6 * J3;
                pt += 4;
                *pt += J7 * J3;
                pt += 4;
                *pt += J8 * J3;
                pt += 4;
                *pt += J9 * J3;
                pt += 4;
                *pt += J10 * J3;
                pt += 4;
                *pt += J11 * J3;
                pt += 4;
                *pt += J12 * J3;
                pt += 4;
                *pt += J13 * J3;
                pt += 4;

                *pt += J4 * J4;
                pt += 4;
                *pt += J5 * J4;
                pt += 4;
                *pt += J6 * J4;
                pt += 4;
                *pt += J7 * J4;
                pt += 4;
                *pt += J8 * J4;
                pt += 4;
                *pt += J9 * J4;
                pt += 4;
                *pt += J10 * J4;
                pt += 4;
                *pt += J11 * J4;
                pt += 4;
                *pt += J12 * J4;
                pt += 4;
                *pt += J13 * J4;
                pt += 4;

                *pt += J5 * J5;
                pt += 4;
                *pt += J6 * J5;
                pt += 4;
                *pt += J7 * J5;
                pt += 4;
                *pt += J8 * J5;
                pt += 4;
                *pt += J9 * J5;
                pt += 4;
                *pt += J10 * J5;
                pt += 4;
                *pt += J11 * J5;
                pt += 4;
                *pt += J12 * J5;
                pt += 4;
                *pt += J13 * J5;
                pt += 4;

                *pt += J6 * J6;
                pt += 4;
                *pt += J7 * J6;
                pt += 4;
                *pt += J8 * J6;
                pt += 4;
                *pt += J9 * J6;
                pt += 4;
                *pt += J10 * J6;
                pt += 4;
                *pt += J11 * J6;
                pt += 4;
                *pt += J12 * J6;
                pt += 4;
                *pt += J13 * J6;
                pt += 4;

                *pt += J7 * J7;
                pt += 4;
                *pt += J8 * J7;
                pt += 4;
                *pt += J9 * J7;
                pt += 4;
                *pt += J10 * J7;
                pt += 4;
                *pt += J11 * J7;
                pt += 4;
                *pt += J12 * J7;
                pt += 4;
                *pt += J13 * J7;
                pt += 4;

                *pt += J8 * J8;
                pt += 4;
                *pt += J9 * J8;
                pt += 4;
                *pt += J10 * J8;
                pt += 4;
                *pt += J11 * J8;
                pt += 4;
                *pt += J12 * J8;
                pt += 4;
                *pt += J13 * J8;
                pt += 4;

                *pt += J9 * J9;
                pt += 4;
                *pt += J10 * J9;
                pt += 4;
                *pt += J11 * J9;
                pt += 4;
                *pt += J12 * J9;
                pt += 4;
                *pt += J13 * J9;
                pt += 4;

                *pt += J10 * J10;
                pt += 4;
                *pt += J11 * J10;
                pt += 4;
                *pt += J12 * J10;
                pt += 4;
                *pt += J13 * J10;
                pt += 4;

                *pt += J11 * J11;
                pt += 4;
                *pt += J12 * J11;
                pt += 4;
                *pt += J13 * J11;
                pt += 4;

                *pt += J12 * J12;
                pt += 4;
                *pt += J13 * J12;
                pt += 4;

                *pt += J13 * J13;
                pt += 4;

                num++;
                numIn1++;
                shiftUp(false);
            }


        private:
            EIGEN_ALIGN16 float SSEData[4 * 105];
            EIGEN_ALIGN16 float SSEData1k[4 * 105];
            EIGEN_ALIGN16 float SSEData1m[4 * 105];
            float numIn1, numIn1k, numIn1m;


            void shiftUp(bool force) {
                if (numIn1 > 1000 || force) {
                    for (int i = 0; i < 105; i++)
                        _mm_store_ps(SSEData1k + 4 * i,
                                     _mm_add_ps(_mm_load_ps(SSEData + 4 * i), _mm_load_ps(SSEData1k + 4 * i)));
                    numIn1k += numIn1;
                    numIn1 = 0;
                    memset(SSEData, 0, sizeof(float) * 4 * 105);
                }

                if (numIn1k > 1000 || force) {
                    for (int i = 0; i < 105; i++)
                        _mm_store_ps(SSEData1m + 4 * i,
                                     _mm_add_ps(_mm_load_ps(SSEData1k + 4 * i), _mm_load_ps(SSEData1m + 4 * i)));
                    numIn1m += numIn1k;
                    numIn1k = 0;
                    memset(SSEData1k, 0, sizeof(float) * 4 * 105);
                }
            }
        };


        /*
         * computes the outer sum of 10x2 matrices, weighted with a 2x2 matrix:
         * 			H = [x y] * [a b; b c] * [x y]^T
         * (assuming x,y are column-vectors).
         * numerically robust to large sums.
         */
        class AccumulatorApprox {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            Mat1313f H;
            size_t num;

            inline void initialize() {
                memset(Data, 0, sizeof(float) * 60);
                memset(Data1k, 0, sizeof(float) * 60);
                memset(Data1m, 0, sizeof(float) * 60);

                memset(TopRight_Data, 0, sizeof(float) * 32);
                memset(TopRight_Data1k, 0, sizeof(float) * 32);
                memset(TopRight_Data1m, 0, sizeof(float) * 32);

                memset(BotRight_Data, 0, sizeof(float) * 8);
                memset(BotRight_Data1k, 0, sizeof(float) * 8);
                memset(BotRight_Data1m, 0, sizeof(float) * 8);
                num = numIn1 = numIn1k = numIn1m = 0;
            }

            inline void finish() {
                H.setZero();
                shiftUp(true);
                assert(numIn1 == 0);
                assert(numIn1k == 0);

                int idx = 0;
                for (int r = 0; r < 10; r++)
                    for (int c = r; c < 10; c++) {
                        H(r, c) = H(c, r) = Data1m[idx];
                        idx++;
                    }

                idx = 0;
                for (int r = 0; r < 10; r++)
                    for (int c = 0; c < 3; c++) {
                        H(r, c + 10) = H(c + 10, r) = TopRight_Data1m[idx];
                        idx++;
                    }

                H(10, 10) = BotRight_Data1m[0];
                H(10, 11) = H(11, 10) = BotRight_Data1m[1];
                H(10, 12) = H(12, 10) = BotRight_Data1m[2];
                H(11, 11) = BotRight_Data1m[3];
                H(11, 12) = H(12, 11) = BotRight_Data1m[4];
                H(12, 12) = BotRight_Data1m[5];


                num = numIn1 + numIn1k + numIn1m;
            }


            inline void updateSSE(
                    const float *const x,
                    const float *const y,
                    const float a,
                    const float b,
                    const float c) {

                Data[0] += a * x[0] * x[0] + c * y[0] * y[0] + b * (x[0] * y[0] + y[0] * x[0]);
                Data[1] += a * x[1] * x[0] + c * y[1] * y[0] + b * (x[1] * y[0] + y[1] * x[0]);
                Data[2] += a * x[2] * x[0] + c * y[2] * y[0] + b * (x[2] * y[0] + y[2] * x[0]);
                Data[3] += a * x[3] * x[0] + c * y[3] * y[0] + b * (x[3] * y[0] + y[3] * x[0]);
                Data[4] += a * x[4] * x[0] + c * y[4] * y[0] + b * (x[4] * y[0] + y[4] * x[0]);
                Data[5] += a * x[5] * x[0] + c * y[5] * y[0] + b * (x[5] * y[0] + y[5] * x[0]);
                Data[6] += a * x[6] * x[0] + c * y[6] * y[0] + b * (x[6] * y[0] + y[6] * x[0]);
                Data[7] += a * x[7] * x[0] + c * y[7] * y[0] + b * (x[7] * y[0] + y[7] * x[0]);
                Data[8] += a * x[8] * x[0] + c * y[8] * y[0] + b * (x[8] * y[0] + y[8] * x[0]);
                Data[9] += a * x[9] * x[0] + c * y[9] * y[0] + b * (x[9] * y[0] + y[9] * x[0]);


                Data[10] += a * x[1] * x[1] + c * y[1] * y[1] + b * (x[1] * y[1] + y[1] * x[1]);
                Data[11] += a * x[2] * x[1] + c * y[2] * y[1] + b * (x[2] * y[1] + y[2] * x[1]);
                Data[12] += a * x[3] * x[1] + c * y[3] * y[1] + b * (x[3] * y[1] + y[3] * x[1]);
                Data[13] += a * x[4] * x[1] + c * y[4] * y[1] + b * (x[4] * y[1] + y[4] * x[1]);
                Data[14] += a * x[5] * x[1] + c * y[5] * y[1] + b * (x[5] * y[1] + y[5] * x[1]);
                Data[15] += a * x[6] * x[1] + c * y[6] * y[1] + b * (x[6] * y[1] + y[6] * x[1]);
                Data[16] += a * x[7] * x[1] + c * y[7] * y[1] + b * (x[7] * y[1] + y[7] * x[1]);
                Data[17] += a * x[8] * x[1] + c * y[8] * y[1] + b * (x[8] * y[1] + y[8] * x[1]);
                Data[18] += a * x[9] * x[1] + c * y[9] * y[1] + b * (x[9] * y[1] + y[9] * x[1]);


                Data[19] += a * x[2] * x[2] + c * y[2] * y[2] + b * (x[2] * y[2] + y[2] * x[2]);
                Data[20] += a * x[3] * x[2] + c * y[3] * y[2] + b * (x[3] * y[2] + y[3] * x[2]);
                Data[21] += a * x[4] * x[2] + c * y[4] * y[2] + b * (x[4] * y[2] + y[4] * x[2]);
                Data[22] += a * x[5] * x[2] + c * y[5] * y[2] + b * (x[5] * y[2] + y[5] * x[2]);
                Data[23] += a * x[6] * x[2] + c * y[6] * y[2] + b * (x[6] * y[2] + y[6] * x[2]);
                Data[24] += a * x[7] * x[2] + c * y[7] * y[2] + b * (x[7] * y[2] + y[7] * x[2]);
                Data[25] += a * x[8] * x[2] + c * y[8] * y[2] + b * (x[8] * y[2] + y[8] * x[2]);
                Data[26] += a * x[9] * x[2] + c * y[9] * y[2] + b * (x[9] * y[2] + y[9] * x[2]);


                Data[27] += a * x[3] * x[3] + c * y[3] * y[3] + b * (x[3] * y[3] + y[3] * x[3]);
                Data[28] += a * x[4] * x[3] + c * y[4] * y[3] + b * (x[4] * y[3] + y[4] * x[3]);
                Data[29] += a * x[5] * x[3] + c * y[5] * y[3] + b * (x[5] * y[3] + y[5] * x[3]);
                Data[30] += a * x[6] * x[3] + c * y[6] * y[3] + b * (x[6] * y[3] + y[6] * x[3]);
                Data[31] += a * x[7] * x[3] + c * y[7] * y[3] + b * (x[7] * y[3] + y[7] * x[3]);
                Data[32] += a * x[8] * x[3] + c * y[8] * y[3] + b * (x[8] * y[3] + y[8] * x[3]);
                Data[33] += a * x[9] * x[3] + c * y[9] * y[3] + b * (x[9] * y[3] + y[9] * x[3]);


                Data[34] += a * x[4] * x[4] + c * y[4] * y[4] + b * (x[4] * y[4] + y[4] * x[4]);
                Data[35] += a * x[5] * x[4] + c * y[5] * y[4] + b * (x[5] * y[4] + y[5] * x[4]);
                Data[36] += a * x[6] * x[4] + c * y[6] * y[4] + b * (x[6] * y[4] + y[6] * x[4]);
                Data[37] += a * x[7] * x[4] + c * y[7] * y[4] + b * (x[7] * y[4] + y[7] * x[4]);
                Data[38] += a * x[8] * x[4] + c * y[8] * y[4] + b * (x[8] * y[4] + y[8] * x[4]);
                Data[39] += a * x[9] * x[4] + c * y[9] * y[4] + b * (x[9] * y[4] + y[9] * x[4]);


                Data[40] += a * x[5] * x[5] + c * y[5] * y[5] + b * (x[5] * y[5] + y[5] * x[5]);
                Data[41] += a * x[6] * x[5] + c * y[6] * y[5] + b * (x[6] * y[5] + y[6] * x[5]);
                Data[42] += a * x[7] * x[5] + c * y[7] * y[5] + b * (x[7] * y[5] + y[7] * x[5]);
                Data[43] += a * x[8] * x[5] + c * y[8] * y[5] + b * (x[8] * y[5] + y[8] * x[5]);
                Data[44] += a * x[9] * x[5] + c * y[9] * y[5] + b * (x[9] * y[5] + y[9] * x[5]);


                Data[45] += a * x[6] * x[6] + c * y[6] * y[6] + b * (x[6] * y[6] + y[6] * x[6]);
                Data[46] += a * x[7] * x[6] + c * y[7] * y[6] + b * (x[7] * y[6] + y[7] * x[6]);
                Data[47] += a * x[8] * x[6] + c * y[8] * y[6] + b * (x[8] * y[6] + y[8] * x[6]);
                Data[48] += a * x[9] * x[6] + c * y[9] * y[6] + b * (x[9] * y[6] + y[9] * x[6]);


                Data[49] += a * x[7] * x[7] + c * y[7] * y[7] + b * (x[7] * y[7] + y[7] * x[7]);
                Data[50] += a * x[8] * x[7] + c * y[8] * y[7] + b * (x[8] * y[7] + y[8] * x[7]);
                Data[51] += a * x[9] * x[7] + c * y[9] * y[7] + b * (x[9] * y[7] + y[9] * x[7]);


                Data[52] += a * x[8] * x[8] + c * y[8] * y[8] + b * (x[8] * y[8] + y[8] * x[8]);
                Data[53] += a * x[9] * x[8] + c * y[9] * y[8] + b * (x[9] * y[8] + y[9] * x[8]);

                Data[54] += a * x[9] * x[9] + c * y[9] * y[9] + b * (x[9] * y[9] + y[9] * x[9]);


                num++;
                numIn1++;
                shiftUp(false);
            }


            /*
             * same as other method, just that x/y are composed of two parts, the first 4 elements are in x4/y4, the last 6 in x6/y6.
             */
            inline void update(
                    const float *const x4,
                    const float *const x6,
                    const float *const y4,
                    const float *const y6,
                    const float a,
                    const float b,
                    const float c) {

                Data[0] += a * x4[0] * x4[0] + c * y4[0] * y4[0] + b * (x4[0] * y4[0] + y4[0] * x4[0]);
                Data[1] += a * x4[1] * x4[0] + c * y4[1] * y4[0] + b * (x4[1] * y4[0] + y4[1] * x4[0]);
                Data[2] += a * x4[2] * x4[0] + c * y4[2] * y4[0] + b * (x4[2] * y4[0] + y4[2] * x4[0]);
                Data[3] += a * x4[3] * x4[0] + c * y4[3] * y4[0] + b * (x4[3] * y4[0] + y4[3] * x4[0]);
                Data[4] += a * x6[0] * x4[0] + c * y6[0] * y4[0] + b * (x6[0] * y4[0] + y6[0] * x4[0]);
                Data[5] += a * x6[1] * x4[0] + c * y6[1] * y4[0] + b * (x6[1] * y4[0] + y6[1] * x4[0]);
                Data[6] += a * x6[2] * x4[0] + c * y6[2] * y4[0] + b * (x6[2] * y4[0] + y6[2] * x4[0]);
                Data[7] += a * x6[3] * x4[0] + c * y6[3] * y4[0] + b * (x6[3] * y4[0] + y6[3] * x4[0]);
                Data[8] += a * x6[4] * x4[0] + c * y6[4] * y4[0] + b * (x6[4] * y4[0] + y6[4] * x4[0]);
                Data[9] += a * x6[5] * x4[0] + c * y6[5] * y4[0] + b * (x6[5] * y4[0] + y6[5] * x4[0]);


                Data[10] += a * x4[1] * x4[1] + c * y4[1] * y4[1] + b * (x4[1] * y4[1] + y4[1] * x4[1]);
                Data[11] += a * x4[2] * x4[1] + c * y4[2] * y4[1] + b * (x4[2] * y4[1] + y4[2] * x4[1]);
                Data[12] += a * x4[3] * x4[1] + c * y4[3] * y4[1] + b * (x4[3] * y4[1] + y4[3] * x4[1]);
                Data[13] += a * x6[0] * x4[1] + c * y6[0] * y4[1] + b * (x6[0] * y4[1] + y6[0] * x4[1]);
                Data[14] += a * x6[1] * x4[1] + c * y6[1] * y4[1] + b * (x6[1] * y4[1] + y6[1] * x4[1]);
                Data[15] += a * x6[2] * x4[1] + c * y6[2] * y4[1] + b * (x6[2] * y4[1] + y6[2] * x4[1]);
                Data[16] += a * x6[3] * x4[1] + c * y6[3] * y4[1] + b * (x6[3] * y4[1] + y6[3] * x4[1]);
                Data[17] += a * x6[4] * x4[1] + c * y6[4] * y4[1] + b * (x6[4] * y4[1] + y6[4] * x4[1]);
                Data[18] += a * x6[5] * x4[1] + c * y6[5] * y4[1] + b * (x6[5] * y4[1] + y6[5] * x4[1]);


                Data[19] += a * x4[2] * x4[2] + c * y4[2] * y4[2] + b * (x4[2] * y4[2] + y4[2] * x4[2]);
                Data[20] += a * x4[3] * x4[2] + c * y4[3] * y4[2] + b * (x4[3] * y4[2] + y4[3] * x4[2]);
                Data[21] += a * x6[0] * x4[2] + c * y6[0] * y4[2] + b * (x6[0] * y4[2] + y6[0] * x4[2]);
                Data[22] += a * x6[1] * x4[2] + c * y6[1] * y4[2] + b * (x6[1] * y4[2] + y6[1] * x4[2]);
                Data[23] += a * x6[2] * x4[2] + c * y6[2] * y4[2] + b * (x6[2] * y4[2] + y6[2] * x4[2]);
                Data[24] += a * x6[3] * x4[2] + c * y6[3] * y4[2] + b * (x6[3] * y4[2] + y6[3] * x4[2]);
                Data[25] += a * x6[4] * x4[2] + c * y6[4] * y4[2] + b * (x6[4] * y4[2] + y6[4] * x4[2]);
                Data[26] += a * x6[5] * x4[2] + c * y6[5] * y4[2] + b * (x6[5] * y4[2] + y6[5] * x4[2]);


                Data[27] += a * x4[3] * x4[3] + c * y4[3] * y4[3] + b * (x4[3] * y4[3] + y4[3] * x4[3]);
                Data[28] += a * x6[0] * x4[3] + c * y6[0] * y4[3] + b * (x6[0] * y4[3] + y6[0] * x4[3]);
                Data[29] += a * x6[1] * x4[3] + c * y6[1] * y4[3] + b * (x6[1] * y4[3] + y6[1] * x4[3]);
                Data[30] += a * x6[2] * x4[3] + c * y6[2] * y4[3] + b * (x6[2] * y4[3] + y6[2] * x4[3]);
                Data[31] += a * x6[3] * x4[3] + c * y6[3] * y4[3] + b * (x6[3] * y4[3] + y6[3] * x4[3]);
                Data[32] += a * x6[4] * x4[3] + c * y6[4] * y4[3] + b * (x6[4] * y4[3] + y6[4] * x4[3]);
                Data[33] += a * x6[5] * x4[3] + c * y6[5] * y4[3] + b * (x6[5] * y4[3] + y6[5] * x4[3]);


                Data[34] += a * x6[0] * x6[0] + c * y6[0] * y6[0] + b * (x6[0] * y6[0] + y6[0] * x6[0]);
                Data[35] += a * x6[1] * x6[0] + c * y6[1] * y6[0] + b * (x6[1] * y6[0] + y6[1] * x6[0]);
                Data[36] += a * x6[2] * x6[0] + c * y6[2] * y6[0] + b * (x6[2] * y6[0] + y6[2] * x6[0]);
                Data[37] += a * x6[3] * x6[0] + c * y6[3] * y6[0] + b * (x6[3] * y6[0] + y6[3] * x6[0]);
                Data[38] += a * x6[4] * x6[0] + c * y6[4] * y6[0] + b * (x6[4] * y6[0] + y6[4] * x6[0]);
                Data[39] += a * x6[5] * x6[0] + c * y6[5] * y6[0] + b * (x6[5] * y6[0] + y6[5] * x6[0]);


                Data[40] += a * x6[1] * x6[1] + c * y6[1] * y6[1] + b * (x6[1] * y6[1] + y6[1] * x6[1]);
                Data[41] += a * x6[2] * x6[1] + c * y6[2] * y6[1] + b * (x6[2] * y6[1] + y6[2] * x6[1]);
                Data[42] += a * x6[3] * x6[1] + c * y6[3] * y6[1] + b * (x6[3] * y6[1] + y6[3] * x6[1]);
                Data[43] += a * x6[4] * x6[1] + c * y6[4] * y6[1] + b * (x6[4] * y6[1] + y6[4] * x6[1]);
                Data[44] += a * x6[5] * x6[1] + c * y6[5] * y6[1] + b * (x6[5] * y6[1] + y6[5] * x6[1]);


                Data[45] += a * x6[2] * x6[2] + c * y6[2] * y6[2] + b * (x6[2] * y6[2] + y6[2] * x6[2]);
                Data[46] += a * x6[3] * x6[2] + c * y6[3] * y6[2] + b * (x6[3] * y6[2] + y6[3] * x6[2]);
                Data[47] += a * x6[4] * x6[2] + c * y6[4] * y6[2] + b * (x6[4] * y6[2] + y6[4] * x6[2]);
                Data[48] += a * x6[5] * x6[2] + c * y6[5] * y6[2] + b * (x6[5] * y6[2] + y6[5] * x6[2]);


                Data[49] += a * x6[3] * x6[3] + c * y6[3] * y6[3] + b * (x6[3] * y6[3] + y6[3] * x6[3]);
                Data[50] += a * x6[4] * x6[3] + c * y6[4] * y6[3] + b * (x6[4] * y6[3] + y6[4] * x6[3]);
                Data[51] += a * x6[5] * x6[3] + c * y6[5] * y6[3] + b * (x6[5] * y6[3] + y6[5] * x6[3]);


                Data[52] += a * x6[4] * x6[4] + c * y6[4] * y6[4] + b * (x6[4] * y6[4] + y6[4] * x6[4]);
                Data[53] += a * x6[5] * x6[4] + c * y6[5] * y6[4] + b * (x6[5] * y6[4] + y6[5] * x6[4]);

                Data[54] += a * x6[5] * x6[5] + c * y6[5] * y6[5] + b * (x6[5] * y6[5] + y6[5] * x6[5]);


                num++;
                numIn1++;
                shiftUp(false);
            }


            inline void updateTopRight(
                    const float *const x4,
                    const float *const x6,
                    const float *const y4,
                    const float *const y6,
                    const float TR00, const float TR10,
                    const float TR01, const float TR11,
                    const float TR02, const float TR12) {
                TopRight_Data[0] += x4[0] * TR00 + y4[0] * TR10;
                TopRight_Data[1] += x4[0] * TR01 + y4[0] * TR11;
                TopRight_Data[2] += x4[0] * TR02 + y4[0] * TR12;

                TopRight_Data[3] += x4[1] * TR00 + y4[1] * TR10;
                TopRight_Data[4] += x4[1] * TR01 + y4[1] * TR11;
                TopRight_Data[5] += x4[1] * TR02 + y4[1] * TR12;

                TopRight_Data[6] += x4[2] * TR00 + y4[2] * TR10;
                TopRight_Data[7] += x4[2] * TR01 + y4[2] * TR11;
                TopRight_Data[8] += x4[2] * TR02 + y4[2] * TR12;

                TopRight_Data[9] += x4[3] * TR00 + y4[3] * TR10;
                TopRight_Data[10] += x4[3] * TR01 + y4[3] * TR11;
                TopRight_Data[11] += x4[3] * TR02 + y4[3] * TR12;

                TopRight_Data[12] += x6[0] * TR00 + y6[0] * TR10;
                TopRight_Data[13] += x6[0] * TR01 + y6[0] * TR11;
                TopRight_Data[14] += x6[0] * TR02 + y6[0] * TR12;

                TopRight_Data[15] += x6[1] * TR00 + y6[1] * TR10;
                TopRight_Data[16] += x6[1] * TR01 + y6[1] * TR11;
                TopRight_Data[17] += x6[1] * TR02 + y6[1] * TR12;

                TopRight_Data[18] += x6[2] * TR00 + y6[2] * TR10;
                TopRight_Data[19] += x6[2] * TR01 + y6[2] * TR11;
                TopRight_Data[20] += x6[2] * TR02 + y6[2] * TR12;

                TopRight_Data[21] += x6[3] * TR00 + y6[3] * TR10;
                TopRight_Data[22] += x6[3] * TR01 + y6[3] * TR11;
                TopRight_Data[23] += x6[3] * TR02 + y6[3] * TR12;

                TopRight_Data[24] += x6[4] * TR00 + y6[4] * TR10;
                TopRight_Data[25] += x6[4] * TR01 + y6[4] * TR11;
                TopRight_Data[26] += x6[4] * TR02 + y6[4] * TR12;

                TopRight_Data[27] += x6[5] * TR00 + y6[5] * TR10;
                TopRight_Data[28] += x6[5] * TR01 + y6[5] * TR11;
                TopRight_Data[29] += x6[5] * TR02 + y6[5] * TR12;

            }

            inline void updateBotRight(
                    const float a00,
                    const float a01,
                    const float a02,
                    const float a11,
                    const float a12,
                    const float a22) {
                BotRight_Data[0] += a00;
                BotRight_Data[1] += a01;
                BotRight_Data[2] += a02;
                BotRight_Data[3] += a11;
                BotRight_Data[4] += a12;
                BotRight_Data[5] += a22;
            }


        private:
            EIGEN_ALIGN16 float Data[60];
            EIGEN_ALIGN16 float Data1k[60];
            EIGEN_ALIGN16 float Data1m[60];

            EIGEN_ALIGN16 float TopRight_Data[32];
            EIGEN_ALIGN16 float TopRight_Data1k[32];
            EIGEN_ALIGN16 float TopRight_Data1m[32];

            EIGEN_ALIGN16 float BotRight_Data[8];
            EIGEN_ALIGN16 float BotRight_Data1k[8];
            EIGEN_ALIGN16 float BotRight_Data1m[8];


            float numIn1, numIn1k, numIn1m;


            void shiftUp(bool force) {
                if (numIn1 > 1000 || force) {
                    for (int i = 0; i < 60; i += 4)
                        _mm_store_ps(Data1k + i, _mm_add_ps(_mm_load_ps(Data + i), _mm_load_ps(Data1k + i)));
                    for (int i = 0; i < 32; i += 4)
                        _mm_store_ps(TopRight_Data1k + i,
                                     _mm_add_ps(_mm_load_ps(TopRight_Data + i), _mm_load_ps(TopRight_Data1k + i)));
                    for (int i = 0; i < 8; i += 4)
                        _mm_store_ps(BotRight_Data1k + i,
                                     _mm_add_ps(_mm_load_ps(BotRight_Data + i), _mm_load_ps(BotRight_Data1k + i)));


                    numIn1k += numIn1;
                    numIn1 = 0;
                    memset(Data, 0, sizeof(float) * 60);
                    memset(TopRight_Data, 0, sizeof(float) * 32);
                    memset(BotRight_Data, 0, sizeof(float) * 8);
                }

                if (numIn1k > 1000 || force) {
                    for (int i = 0; i < 60; i += 4)
                        _mm_store_ps(Data1m + i, _mm_add_ps(_mm_load_ps(Data1k + i), _mm_load_ps(Data1m + i)));
                    for (int i = 0; i < 32; i += 4)
                        _mm_store_ps(TopRight_Data1m + i,
                                     _mm_add_ps(_mm_load_ps(TopRight_Data1k + i), _mm_load_ps(TopRight_Data1m + i)));
                    for (int i = 0; i < 8; i += 4)
                        _mm_store_ps(BotRight_Data1m + i,
                                     _mm_add_ps(_mm_load_ps(BotRight_Data1k + i), _mm_load_ps(BotRight_Data1m + i)));

                    numIn1m += numIn1k;
                    numIn1k = 0;
                    memset(Data1k, 0, sizeof(float) * 60);
                    memset(TopRight_Data1k, 0, sizeof(float) * 32);
                    memset(BotRight_Data1k, 0, sizeof(float) * 8);
                }
            }
        };


        // multi-fisheye
        // 用于滑动窗口优化 非Schur补部分
        class AccumulatorNoschur
        {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            Mat5757f H;
            size_t num;

            inline void initialize() 
            {
                memset(Data, 0, sizeof(float) * 24);
                memset(Data1k, 0, sizeof(float) * 24);
                memset(Data1m, 0, sizeof(float) * 24);

                memset(TopRight_Data, 0, sizeof(float) * 308);
                memset(TopRight_Data1k, 0, sizeof(float) * 308);
                memset(TopRight_Data1m, 0, sizeof(float) * 308);

                memset(BotRight_Data, 0, sizeof(float) * 1328);
                memset(BotRight_Data1k, 0, sizeof(float) * 1328);
                memset(BotRight_Data1m, 0, sizeof(float) * 1328);
                num = numIn1 = numIn1k = numIn1m = 0;

                mappingBR.setZero();
                int n1 = 0;
                for(int i = 0; i <51; i++)
                {
                    for(int j = i; j < 51; j++)
                    {
                        mappingBR(i, j) = n1;
                        n1++;
                    }
                }
                mappingTR.setZero();
                int n2 = 0;
                for(int i = 0; i <6; i++)
                {
                    for(int j = 0; j < 51; j++)
                    {
                        mappingTR(i, j) = n2;
                        n2++;
                    }
                }
            }

            inline void finish() 
            {
                H.setZero();
                shiftUp(true);
                assert(numIn1 == 0);
                assert(numIn1k == 0);

                int idx = 0;
                for (int r = 0; r < 6; r++)
                    for (int c = r; c < 6; c++) 
                    {
                        H(r, c) = H(c, r) = Data1m[idx];
                        idx++;
                    }

                idx = 0;
                for (int r = 0; r < 6; r++)
                    for (int c = 0; c < 51; c++) 
                    {
                        H(r, c + 6) = H(c + 6, r) = TopRight_Data1m[idx];
                        idx++;
                    }

                idx = 0;
                for (int r = 0; r < 51; r++)
                    for (int c = r; c < 51; c++) 
                    {
                        H(r + 6, c + 6) = H(c + 6, r + 6) = BotRight_Data1m[idx];
                        idx++;
                    }


                num = numIn1 + numIn1k + numIn1m;
            }

            inline void update(
                    const float *const x6,
                    const float *const y6,
                    const float a,
                    const float b,
                    const float c) 
            {
            
                Data[0] += a * x6[0] * x6[0] + c * y6[0] * y6[0] + b * (x6[0] * y6[0] + y6[0] * x6[0]);
                Data[1] += a * x6[1] * x6[0] + c * y6[1] * y6[0] + b * (x6[1] * y6[0] + y6[1] * x6[0]);
                Data[2] += a * x6[2] * x6[0] + c * y6[2] * y6[0] + b * (x6[2] * y6[0] + y6[2] * x6[0]);
                Data[3] += a * x6[3] * x6[0] + c * y6[3] * y6[0] + b * (x6[3] * y6[0] + y6[3] * x6[0]);
                Data[4] += a * x6[4] * x6[0] + c * y6[4] * y6[0] + b * (x6[4] * y6[0] + y6[4] * x6[0]);
                Data[5] += a * x6[5] * x6[0] + c * y6[5] * y6[0] + b * (x6[5] * y6[0] + y6[5] * x6[0]);


                Data[6] += a * x6[1] * x6[1] + c * y6[1] * y6[1] + b * (x6[1] * y6[1] + y6[1] * x6[1]);
                Data[7] += a * x6[2] * x6[1] + c * y6[2] * y6[1] + b * (x6[2] * y6[1] + y6[2] * x6[1]);
                Data[8] += a * x6[3] * x6[1] + c * y6[3] * y6[1] + b * (x6[3] * y6[1] + y6[3] * x6[1]);
                Data[9] += a * x6[4] * x6[1] + c * y6[4] * y6[1] + b * (x6[4] * y6[1] + y6[4] * x6[1]);
                Data[10] += a * x6[5] * x6[1] + c * y6[5] * y6[1] + b * (x6[5] * y6[1] + y6[5] * x6[1]);


                Data[11] += a * x6[2] * x6[2] + c * y6[2] * y6[2] + b * (x6[2] * y6[2] + y6[2] * x6[2]);
                Data[12] += a * x6[3] * x6[2] + c * y6[3] * y6[2] + b * (x6[3] * y6[2] + y6[3] * x6[2]);
                Data[13] += a * x6[4] * x6[2] + c * y6[4] * y6[2] + b * (x6[4] * y6[2] + y6[4] * x6[2]);
                Data[14] += a * x6[5] * x6[2] + c * y6[5] * y6[2] + b * (x6[5] * y6[2] + y6[5] * x6[2]);


                Data[15] += a * x6[3] * x6[3] + c * y6[3] * y6[3] + b * (x6[3] * y6[3] + y6[3] * x6[3]);
                Data[16] += a * x6[4] * x6[3] + c * y6[4] * y6[3] + b * (x6[4] * y6[3] + y6[4] * x6[3]);
                Data[17] += a * x6[5] * x6[3] + c * y6[5] * y6[3] + b * (x6[5] * y6[3] + y6[5] * x6[3]);


                Data[18] += a * x6[4] * x6[4] + c * y6[4] * y6[4] + b * (x6[4] * y6[4] + y6[4] * x6[4]);
                Data[19] += a * x6[5] * x6[4] + c * y6[5] * y6[4] + b * (x6[5] * y6[4] + y6[5] * x6[4]);

                Data[20] += a * x6[5] * x6[5] + c * y6[5] * y6[5] + b * (x6[5] * y6[5] + y6[5] * x6[5]);
                num++;
                numIn1++;
                shiftUp(false);
            }

            inline void updateTopRight(
                    const float *const x6,
                    const float *const y6,
                    const float TR00, const float TR10,
                    const float TR01, const float TR11,
                    const float TR02, const float TR12, 
                    const int hcamnum, const float tcamnum)
            {
                int col = 2 * (hcamnum * 5 +tcamnum);
                TopRight_Data[mappingTR(0,col)] += x6[0] * TR00 + y6[0] * TR10;
                TopRight_Data[mappingTR(0,col+1)] += x6[0] * TR01 + y6[0] * TR11;
                TopRight_Data[mappingTR(0,50)] += x6[0] * TR02 + y6[0] * TR12;

                TopRight_Data[mappingTR(1,col)] += x6[1] * TR00 + y6[1] * TR10;
                TopRight_Data[mappingTR(1,col+1)] += x6[1] * TR01 + y6[1] * TR11;
                TopRight_Data[mappingTR(1,50)] += x6[1] * TR02 + y6[1] * TR12;

                TopRight_Data[mappingTR(2,col)] += x6[2] * TR00 + y6[2] * TR10;
                TopRight_Data[mappingTR(2,col+1)] += x6[2] * TR01 + y6[2] * TR11;
                TopRight_Data[mappingTR(2,50)] += x6[2] * TR02 + y6[2] * TR12;

                TopRight_Data[mappingTR(3,col)] += x6[3] * TR00 + y6[3] * TR10;
                TopRight_Data[mappingTR(3,col+1)] += x6[3] * TR01 + y6[3] * TR11;
                TopRight_Data[mappingTR(3,50)] += x6[3] * TR02 + y6[3] * TR12;

                TopRight_Data[mappingTR(4,col)] += x6[4] * TR00 + y6[4] * TR10;
                TopRight_Data[mappingTR(4,col+1)] += x6[4] * TR01 + y6[4] * TR11;
                TopRight_Data[mappingTR(4,50)] += x6[4] * TR02 + y6[4] * TR12;

                TopRight_Data[mappingTR(5,col)] += x6[5] * TR00 + y6[5] * TR10;
                TopRight_Data[mappingTR(5,col+1)] += x6[5] * TR01 + y6[5] * TR11;
                TopRight_Data[mappingTR(5,50)] += x6[5] * TR02 + y6[5] * TR12;
            }

            inline void updateBotRight(
                    const float a00,
                    const float a01,
                    const float a02,
                    const float a11,
                    const float a12,
                    const float a22,
                    const int hcamnum, const float tcamnum) 
            {
                int col = 2 * (hcamnum * 5 +tcamnum);
                BotRight_Data[mappingBR(col, col)] += a00; 
                BotRight_Data[mappingBR(col, col+1)] += a01; 
                BotRight_Data[mappingBR(col+1, col+1)] += a11; 
                BotRight_Data[mappingBR(col, 50)] += a02;
                BotRight_Data[mappingBR(col+1, 50)] += a12;
                BotRight_Data[1326] += a22;
            }

            Eigen::Matrix<int, 51, 51> mappingBR;   // 映射矩阵，便于查找 2维 -> 1维
            Eigen::Matrix<int, 6, 51> mappingTR;    // 映射矩阵，便于查找 2维 -> 1维


        private:
            //! 左上角6*6, 21个值(有对称) 为了sse指令集 内存对齐设置24
            EIGEN_ALIGN16 float Data[24];
            EIGEN_ALIGN16 float Data1k[24];
            EIGEN_ALIGN16 float Data1m[24];
            //! 右上角6*51, 306个值  为了sse指令集 内存对齐设置308
            EIGEN_ALIGN16 float TopRight_Data[308];
            EIGEN_ALIGN16 float TopRight_Data1k[308];
            EIGEN_ALIGN16 float TopRight_Data1m[308];
            //! 右下角51*51, 1326个值(有对称) 为了sse指令集 内存对齐设置1328
            EIGEN_ALIGN16 float BotRight_Data[1328];
            EIGEN_ALIGN16 float BotRight_Data1k[1328];
            EIGEN_ALIGN16 float BotRight_Data1m[1328];
            

            float numIn1, numIn1k, numIn1m;

            void shiftUp(bool force) {
                if (numIn1 > 1000 || force) {
                    for (int i = 0; i < 24; i += 4)
                        _mm_store_ps(Data1k + i, _mm_add_ps(_mm_load_ps(Data + i), _mm_load_ps(Data1k + i)));
                    for (int i = 0; i < 308; i += 4)
                        _mm_store_ps(TopRight_Data1k + i,
                                     _mm_add_ps(_mm_load_ps(TopRight_Data + i), _mm_load_ps(TopRight_Data1k + i)));
                    for (int i = 0; i < 1328; i += 4)
                        _mm_store_ps(BotRight_Data1k + i,
                                     _mm_add_ps(_mm_load_ps(BotRight_Data + i), _mm_load_ps(BotRight_Data1k + i)));


                    numIn1k += numIn1;
                    numIn1 = 0;
                    memset(Data, 0, sizeof(float) * 24);
                    memset(TopRight_Data, 0, sizeof(float) * 308);
                    memset(BotRight_Data, 0, sizeof(float) * 1328);
                }

                if (numIn1k > 1000 || force) {
                    for (int i = 0; i < 24; i += 4)
                        _mm_store_ps(Data1m + i, _mm_add_ps(_mm_load_ps(Data1k + i), _mm_load_ps(Data1m + i)));
                    for (int i = 0; i < 308; i += 4)
                        _mm_store_ps(TopRight_Data1m + i,
                                     _mm_add_ps(_mm_load_ps(TopRight_Data1k + i), _mm_load_ps(TopRight_Data1m + i)));
                    for (int i = 0; i < 1328; i += 4)
                        _mm_store_ps(BotRight_Data1m + i,
                                     _mm_add_ps(_mm_load_ps(BotRight_Data1k + i), _mm_load_ps(BotRight_Data1m + i)));

                    numIn1m += numIn1k;
                    numIn1k = 0;
                    memset(Data1k, 0, sizeof(float) * 24);
                    memset(TopRight_Data1k, 0, sizeof(float) *308);
                    memset(BotRight_Data1k, 0, sizeof(float) * 1328);
                }
            }

        };



        class Accumulator9 {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            Mat99f H;
            Vec9f b;
            size_t num;

            inline void initialize() {
                H.setZero();
                b.setZero();
                memset(SSEData, 0, sizeof(float) * 4 * 45);
                memset(SSEData1k, 0, sizeof(float) * 4 * 45);
                memset(SSEData1m, 0, sizeof(float) * 4 * 45);
                num = numIn1 = numIn1k = numIn1m = 0;
            }
            // finish函数就是强制shiftUp后，全部数据累加到最上层，最后取得数据也就是1m的数据
            inline void finish() {
                H.setZero();
                shiftUp(true);
                assert(numIn1 == 0);
                assert(numIn1k == 0);

                int idx = 0;
                for (int r = 0; r < 9; r++)
                    for (int c = r; c < 9; c++) {
                        float d = SSEData1m[idx + 0] + SSEData1m[idx + 1] + SSEData1m[idx + 2] + SSEData1m[idx + 3];
                        H(r, c) = H(c, r) = d;
                        idx += 4;
                    }
                assert(idx == 4 * 45);
            }

            // updateSSE就是加速更新。
            inline void updateSSE(
                    const __m128 J0, const __m128 J1,
                    const __m128 J2, const __m128 J3,
                    const __m128 J4, const __m128 J5,
                    const __m128 J6, const __m128 J7,
                    const __m128 J8) {
                float *pt = SSEData;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J0)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J1)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J2)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J3)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J4)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J5)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J6)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J7)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J8)));
                pt += 4;

                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J1)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J2)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J3)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J4)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J5)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J6)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J7)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J8)));
                pt += 4;

                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J2)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J3)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J4)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J5)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J6)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J7)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J8)));
                pt += 4;

                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J3)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J4)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J5)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J6)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J7)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J8)));
                pt += 4;

                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J4)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J5)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J6)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J7)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J8)));
                pt += 4;

                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J5)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J6)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J7)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J8)));
                pt += 4;

                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J6)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J7)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J8)));
                pt += 4;

                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7, J7)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7, J8)));
                pt += 4;

                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J8, J8)));
                pt += 4;

                num += 4;
                numIn1++;
                shiftUp(false);
            }


            inline void updateSSE_eighted(
                    const __m128 J0, const __m128 J1,
                    const __m128 J2, const __m128 J3,
                    const __m128 J4, const __m128 J5,
                    const __m128 J6, const __m128 J7,
                    const __m128 J8, const __m128 w) {
                float *pt = SSEData;

                __m128 J0w = _mm_mul_ps(J0, w);
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J0)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J1)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J2)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J3)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J4)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J5)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J6)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J7)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J8)));
                pt += 4;

                __m128 J1w = _mm_mul_ps(J1, w);
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J1)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J2)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J3)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J4)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J5)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J6)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J7)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J8)));
                pt += 4;

                __m128 J2w = _mm_mul_ps(J2, w);
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2w, J2)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2w, J3)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2w, J4)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2w, J5)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2w, J6)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2w, J7)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2w, J8)));
                pt += 4;

                __m128 J3w = _mm_mul_ps(J3, w);
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3w, J3)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3w, J4)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3w, J5)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3w, J6)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3w, J7)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3w, J8)));
                pt += 4;

                __m128 J4w = _mm_mul_ps(J4, w);
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4w, J4)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4w, J5)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4w, J6)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4w, J7)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4w, J8)));
                pt += 4;

                __m128 J5w = _mm_mul_ps(J5, w);
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5w, J5)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5w, J6)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5w, J7)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5w, J8)));
                pt += 4;

                __m128 J6w = _mm_mul_ps(J6, w);
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6w, J6)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6w, J7)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6w, J8)));
                pt += 4;

                __m128 J7w = _mm_mul_ps(J7, w);
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7w, J7)));
                pt += 4;
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7w, J8)));
                pt += 4;

                __m128 J8w = _mm_mul_ps(J8, w);
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J8w, J8)));
                pt += 4;

                num += 4;
                numIn1++;
                shiftUp(false);
            }

            // updateSingle函数就是普通更新，
            inline void updateSingle(
                    const float J0, const float J1,
                    const float J2, const float J3,
                    const float J4, const float J5,
                    const float J6, const float J7,
                    const float J8, int off = 0) {
                float *pt = SSEData + off;
                *pt += J0 * J0;
                pt += 4;
                *pt += J1 * J0;
                pt += 4;
                *pt += J2 * J0;
                pt += 4;
                *pt += J3 * J0;
                pt += 4;
                *pt += J4 * J0;
                pt += 4;
                *pt += J5 * J0;
                pt += 4;
                *pt += J6 * J0;
                pt += 4;
                *pt += J7 * J0;
                pt += 4;
                *pt += J8 * J0;
                pt += 4;


                *pt += J1 * J1;
                pt += 4;
                *pt += J2 * J1;
                pt += 4;
                *pt += J3 * J1;
                pt += 4;
                *pt += J4 * J1;
                pt += 4;
                *pt += J5 * J1;
                pt += 4;
                *pt += J6 * J1;
                pt += 4;
                *pt += J7 * J1;
                pt += 4;
                *pt += J8 * J1;
                pt += 4;


                *pt += J2 * J2;
                pt += 4;
                *pt += J3 * J2;
                pt += 4;
                *pt += J4 * J2;
                pt += 4;
                *pt += J5 * J2;
                pt += 4;
                *pt += J6 * J2;
                pt += 4;
                *pt += J7 * J2;
                pt += 4;
                *pt += J8 * J2;
                pt += 4;


                *pt += J3 * J3;
                pt += 4;
                *pt += J4 * J3;
                pt += 4;
                *pt += J5 * J3;
                pt += 4;
                *pt += J6 * J3;
                pt += 4;
                *pt += J7 * J3;
                pt += 4;
                *pt += J8 * J3;
                pt += 4;


                *pt += J4 * J4;
                pt += 4;
                *pt += J5 * J4;
                pt += 4;
                *pt += J6 * J4;
                pt += 4;
                *pt += J7 * J4;
                pt += 4;
                *pt += J8 * J4;
                pt += 4;

                *pt += J5 * J5;
                pt += 4;
                *pt += J6 * J5;
                pt += 4;
                *pt += J7 * J5;
                pt += 4;
                *pt += J8 * J5;
                pt += 4;


                *pt += J6 * J6;
                pt += 4;
                *pt += J7 * J6;
                pt += 4;
                *pt += J8 * J6;
                pt += 4;


                *pt += J7 * J7;
                pt += 4;
                *pt += J8 * J7;
                pt += 4;

                *pt += J8 * J8;
                pt += 4;

                num++;
                numIn1++;
                shiftUp(false);
            }

            inline void updateSingleWeighted(
                    float J0, float J1,
                    float J2, float J3,
                    float J4, float J5,
                    float J6, float J7,
                    float J8, float w,
                    int off = 0) {

                float *pt = SSEData + off;
                *pt += J0 * J0 * w;
                pt += 4;
                J0 *= w;
                *pt += J1 * J0;
                pt += 4;
                *pt += J2 * J0;
                pt += 4;
                *pt += J3 * J0;
                pt += 4;
                *pt += J4 * J0;
                pt += 4;
                *pt += J5 * J0;
                pt += 4;
                *pt += J6 * J0;
                pt += 4;
                *pt += J7 * J0;
                pt += 4;
                *pt += J8 * J0;
                pt += 4;


                *pt += J1 * J1 * w;
                pt += 4;
                J1 *= w;
                *pt += J2 * J1;
                pt += 4;
                *pt += J3 * J1;
                pt += 4;
                *pt += J4 * J1;
                pt += 4;
                *pt += J5 * J1;
                pt += 4;
                *pt += J6 * J1;
                pt += 4;
                *pt += J7 * J1;
                pt += 4;
                *pt += J8 * J1;
                pt += 4;


                *pt += J2 * J2 * w;
                pt += 4;
                J2 *= w;
                *pt += J3 * J2;
                pt += 4;
                *pt += J4 * J2;
                pt += 4;
                *pt += J5 * J2;
                pt += 4;
                *pt += J6 * J2;
                pt += 4;
                *pt += J7 * J2;
                pt += 4;
                *pt += J8 * J2;
                pt += 4;


                *pt += J3 * J3 * w;
                pt += 4;
                J3 *= w;
                *pt += J4 * J3;
                pt += 4;
                *pt += J5 * J3;
                pt += 4;
                *pt += J6 * J3;
                pt += 4;
                *pt += J7 * J3;
                pt += 4;
                *pt += J8 * J3;
                pt += 4;


                *pt += J4 * J4 * w;
                pt += 4;
                J4 *= w;
                *pt += J5 * J4;
                pt += 4;
                *pt += J6 * J4;
                pt += 4;
                *pt += J7 * J4;
                pt += 4;
                *pt += J8 * J4;
                pt += 4;

                *pt += J5 * J5 * w;
                pt += 4;
                J5 *= w;
                *pt += J6 * J5;
                pt += 4;
                *pt += J7 * J5;
                pt += 4;
                *pt += J8 * J5;
                pt += 4;


                *pt += J6 * J6 * w;
                pt += 4;
                J6 *= w;
                *pt += J7 * J6;
                pt += 4;
                *pt += J8 * J6;
                pt += 4;


                *pt += J7 * J7 * w;
                pt += 4;
                J7 *= w;
                *pt += J8 * J7;
                pt += 4;

                *pt += J8 * J8 * w;
                pt += 4;

                num++;
                numIn1++;
                shiftUp(false);
            }

        // 一个累积器有三层缓存如下所示。SSE的话每个缓存器的大小是4*(1+2+…+X)，比如acc9就是4*45，普通的话没有4×
        private:
            EIGEN_ALIGN16 float SSEData[4 * 45];  // 这里45是 1+2+3+4+5+..+9 = 45 这是由于Hessian是对称阵。
            EIGEN_ALIGN16 float SSEData1k[4 * 45];
            EIGEN_ALIGN16 float SSEData1m[4 * 45];
            float numIn1, numIn1k, numIn1m;

            // shiftUp 向上级存储器转换， 每次update后都检查一次，如果低级缓存数量过大或force=true，数值付给高层并且清空低层
            void shiftUp(bool force) {
                if (numIn1 > 1000 || force) {
                    for (int i = 0; i < 45; i++)
                        _mm_store_ps(SSEData1k + 4 * i,
                                     _mm_add_ps(_mm_load_ps(SSEData + 4 * i), _mm_load_ps(SSEData1k + 4 * i)));
                    numIn1k += numIn1;
                    numIn1 = 0;
                    memset(SSEData, 0, sizeof(float) * 4 * 45);
                }

                if (numIn1k > 1000 || force) {
                    for (int i = 0; i < 45; i++)
                        _mm_store_ps(SSEData1m + 4 * i,
                                     _mm_add_ps(_mm_load_ps(SSEData1k + 4 * i), _mm_load_ps(SSEData1m + 4 * i)));
                    numIn1m += numIn1k;
                    numIn1k = 0;
                    memset(SSEData1k, 0, sizeof(float) * 4 * 45);
                }
            }
        };

        // 用于5镜头 组合全景    6 + 2*5*5 + 1 = 57
        class Accumulator57
        {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            Mat5757f H;
            Vec57f b;
            size_t num;

            inline void initialize() 
            {
                H.setZero();
                b.setZero();
                memset(SSEData, 0, sizeof(float) * 4 * 1653);   // 会对128位, 16字节进行对齐, 因此每个数用4个float存
                memset(SSEData1k, 0, sizeof(float) * 4 * 1653);
                memset(SSEData1m, 0, sizeof(float) * 4 * 1653);
                num = numIn1 = numIn1k = numIn1m = 0;
            }

            // finish函数就是强制shiftUp后，全部数据累加到最上层，最后取得数据也就是1m的数据
            inline void finish() {
                H.setZero();
                shiftUp(true);
                assert(numIn1 == 0);
                assert(numIn1k == 0);

                int idx = 0;
                for (int r = 0; r < 57; r++)
                    for (int c = r; c < 57; c++) {
                        float d = SSEData1m[idx + 0] + SSEData1m[idx + 1] + SSEData1m[idx + 2] + SSEData1m[idx + 3];
                        H(r, c) = H(c, r) = d;
                        idx += 4;
                    }
                assert(idx == 4 * 1653);
            }

            inline void updateSSE(const __m128 J0, const __m128 J1,
                    const __m128 J2, const __m128 J3,
                    const __m128 J4, const __m128 J5,
                    const __m128 J6, const __m128 J7,
                    const __m128 J8, int cam1, int cam2)
            {
                //vector<__m128> J{57, _mm_load_ps((float *) 0)};
                __m128 a = _mm_setzero_ps();
                vector<__m128> J(57, a);
                int n = (cam1 * 5 + cam2) * 2;
                J[0] = J0;
                J[1] = J1;
                J[2] = J2;
                J[3] = J3;
                J[4] = J4;
                J[5] = J5;
                J[n+1 + 5] = J6;
                J[n+2 + 5] = J7;
                J[56] = J8;  //r

                float *pt = SSEData;
                for(int i = 0; i < 57; i++ )
                {
                    for(int j = i; j < 57 ; j++ )
                    {
                        _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J[i], J[j])));
                        pt += 4;
                    }
                }

                num += 4;
                numIn1++;
                shiftUp(false);

            }

            inline void updateSSE_weighted(const __m128 J0, const __m128 J1,
                    const __m128 J2, const __m128 J3,
                    const __m128 J4, const __m128 J5,
                    const __m128 J6, const __m128 J7,
                    const __m128 J8, const __m128 w, const __m128i cam1, const __m128i cam2)
            {
                __m128i cn = _mm_set1_epi32(5);
                __m128i n = _mm_mul_epi32(_mm_add_epi32(_mm_mul_epi32(cam1, cn), cam2), _mm_set1_epi32(2));
                __m128  a = _mm_setzero_ps();
                vector<__m128> J(57, a);
                J[0] = J0;
                J[1] = J1;
                J[2] = J2;
                J[3] = J3;
                J[4] = J4;
                J[5] = J5;
                J[n[0]+1 + 5][0] = J6[0];
                J[n[0]+2 + 5][0] = J7[0];
                J[n[1]+1 + 5][1] = J6[1];
                J[n[1]+2 + 5][1] = J7[1];
                J[n[2]+1 + 5][2] = J6[2];
                J[n[2]+2 + 5][2] = J7[2];
                J[n[3]+1 + 5][3] = J6[3];
                J[n[3]+2 + 5][3] = J7[3];
                J[56] = J8;  //r
                
                float *pt = SSEData;

                for(int i = 0; i < 57; i++ )
                {
                    __m128 Jiw = _mm_mul_ps(J[i], w);
                    for(int j = i; j < 57 ; j++ )
                    {
                        _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(Jiw, J[j])));
                        pt += 4;
                    }
                }

                num += 4;
                numIn1++;
                shiftUp(false);


            }

            inline void updateSingle(const float J0, const float J1,
                    const float J2, const float J3,
                    const float J4, const float J5,
                    const float J6, const float J7,
                    const float J8, const int cam1, const int cam2, int off = 0)
            {
                float *pt = SSEData + off;

                int n = (cam1 * 5 + cam2) * 2;
                vector<float> J(57,  0);
                J[0] = J0;
                J[1] = J1;
                J[2] = J2;
                J[3] = J3;
                J[4] = J4;
                J[5] = J5;
                J[n+6] = J6;
                J[n+7] = J7;
                J[56] = J8;  //r

                for(int i = 0; i < 57; i++ )
                {
                    for(int j = i; j < 57; j++ )
                    {
                        *pt += J[i] * J[j];
                        pt += 4;
                    }
                }

                num++;
                numIn1++;
                shiftUp(false);
            }

            inline void updateSingleWeighted(float J0, float J1,
                    float J2, float J3,
                    float J4, float J5,
                    float J6, float J7,
                    float J8, float w,
                    int cam1, int cam2,
                    int off = 0)
            {
                float *pt = SSEData + off;

                int n = (cam1 * 5 + cam2) * 2;
                vector<float> J(57,  0);
                J[0] = J0;
                J[1] = J1;
                J[2] = J2;
                J[3] = J3;
                J[4] = J4;
                J[5] = J5;
                J[n+1 + 5] = J6;
                J[n+2 + 5] = J7;
                J[56] = J8;  //r

                for(int i = 0; i < 57; i++ )
                {
                    float Jiw = J[i]*w;
                    for(int j = i; j < 57; j++ )
                    {
                        *pt += Jiw * J[j];
                        pt += 4;
                    }
                }

                num++;
                numIn1++;
                shiftUp(false);
            }

        private:
            EIGEN_ALIGN16 float SSEData[4 * 1653];  // 这里1653是 1+2+3+4+5+..+57 = 1653 这是由于Hessian是对称阵。
            EIGEN_ALIGN16 float SSEData1k[4 * 1653];
            EIGEN_ALIGN16 float SSEData1m[4 * 1653];
            float numIn1, numIn1k, numIn1m;

            // shiftUp 向上级存储器转换， 每次update后都检查一次，如果低级缓存数量过大或force=true，数值付给高层并且清空低层
            void shiftUp(bool force) {
                if (numIn1 > 1000 || force) {
                    for (int i = 0; i < 1653; i++)
                        _mm_store_ps(SSEData1k + 4 * i,
                                     _mm_add_ps(_mm_load_ps(SSEData + 4 * i), _mm_load_ps(SSEData1k + 4 * i)));
                    numIn1k += numIn1;
                    numIn1 = 0;
                    memset(SSEData, 0, sizeof(float) * 4 * 1653);
                }

                if (numIn1k > 1000 || force) {
                    for (int i = 0; i < 1653; i++)
                        _mm_store_ps(SSEData1m + 4 * i,
                                     _mm_add_ps(_mm_load_ps(SSEData1k + 4 * i), _mm_load_ps(SSEData1m + 4 * i)));
                    numIn1m += numIn1k;
                    numIn1k = 0;
                    memset(SSEData1k, 0, sizeof(float) * 4 * 1653);
                }
            }
        };

    }
}

#endif //