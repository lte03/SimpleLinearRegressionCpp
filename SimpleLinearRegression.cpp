#include "SimpleLinearRegression.h"
#include <random>
#include <iomanip>
#include <cmath>

namespace ML
{
    SimpleLinearRegression::SimpleLinearRegression(bool randW = false)
    {
        std::setprecision(2);
        this->learningRate = 0.01;
        if(!randW)
        {
            this->w0 = 0;
            this->w1 = 0;
        }
        else
        {
            std::random_device rd;
            std::mt19937 generator(rd());
            std::uniform_real_distribution<> dist(-1,1);
            this->w0 = dist(generator);
            this->w1 = dist(generator);
        }
    }

    inline double SimpleLinearRegression::getW0() const
    {
        return this->w0;
    }

    inline double SimpleLinearRegression::getW1() const
    {
        return this->w1;
    }

    inline void SimpleLinearRegression::setW0(double w0)
    {
        this->w0 = w0;
    }

    inline void SimpleLinearRegression::setW1(double w1)
    {
        this->w1 = w1;
    }

    void SimpleLinearRegression::train(std::vector<double> X,std::vector<double> y)
    {
        if (X.size() != y.size())
        {
            throw -1;
        }
        const int epoch = 1000;
        const int size = y.size();
        for(int i = 0;i < epoch;i++)
        {
            std::vector<double> temp(size);
            double tempW0;
            double tempW1;
            for(int j = 0;j < size;j++)
            {
                temp[j] = this->w0 + (this->w1 * X[j]);
            }
            double dw0 = 0;
            double dw1 = 0;
            for(int j = 0;j < size;j++)
            {
                dw0 += temp[j] - y[j];
                dw1 += (temp[j] - y[j]) * X[j];
            }
            dw0 /= size;
            dw1 /= size;
            this->w0 -= this->learningRate * dw0;
            this->w1 -= this->learningRate * dw1;
        }
    }

    std::vector<double> SimpleLinearRegression::predict(std::vector<double> X)
    {
        std::vector<double> predicts;
        for (int i = 0;i < X.size();i++)
        {
            predicts.push_back(this->w0 + (this->w1 * X[i]));
        }
        return predicts;
    }

    SimpleLinearRegression::~SimpleLinearRegression()
    {

    }
}
