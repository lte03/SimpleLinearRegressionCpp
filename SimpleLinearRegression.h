#ifndef SIMPLE_LR_
#define SIMPLE_LR_

#include <vector>

namespace ML
{
    class SimpleLinearRegression
    {
    private:
        double w0;
        double w1;
        double learningRate;
    public:
        SimpleLinearRegression();
        SimpleLinearRegression(bool rand);
        ~SimpleLinearRegression();
        inline double getW0() const;
        inline double getW1() const;
        inline void setW0(double w0);
        inline void setW1(double w1);
        void train(std::vector<double> X,std::vector<double> y);
        std::vector<double> predict(std::vector<double> X);
    };
}

#endif
