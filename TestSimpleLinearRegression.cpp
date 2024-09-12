#include <iostream>
#include "SimpleLinearRegression.h"

int main()
{
    std::vector<double> X = {1,3,5,8,9};
    std::vector<double> y = {1.2,3.5,5.1,8.2,9.0};
    ML::SimpleLinearRegression lr(false);
    lr.train(X,y);
    std::vector<double> predicts = lr.predict(X);
    for(int i = 0;i < predicts.size();i++)
    {
        std::cout << predicts[i] << std::endl;
    }
    return 0;
}