#include <iostream>
using namespace std;
    
const int num_sec_levels = 4;
const int num_key_format = 1;
    
typedef struct v0curves
{
    int securityLevel;
    double  linearTerm1;
    double  linearTerm2;
    int nAlpha;
    int keyFormat;

    v0curves(    int securityLevel_,
                 double linearTerm1_,
                 double linearTerm2_,
                 int nAlpha_,
                 int keyFormat_)
    {
        securityLevel = securityLevel_;
        linearTerm1 = linearTerm1_;
        linearTerm2 = linearTerm2_;
        nAlpha = nAlpha_;
        keyFormat = keyFormat_;
    }

} v0curves; 
    
v0curves parameters[num_sec_levels][num_key_format] = { 
    {v0curves(1, 4.13213, 7.123123, 1, 1)},
    {v0curves(2, 5.123123, 8.123123, 1, 2)},
    {v0curves(3, 6.123123, 9.1231223, 1, 3)},
    {v0curves(4, 10.1231, 10.123123, 1, 4)}
};
    
extern "C" v0curves *security_estimator(int securityLevel, int keyFormat)
{
    if (securityLevel == 80 ){
        return &parameters[0][keyFormat];
    }
    else if (securityLevel == 128 ){
        return &parameters[1][keyFormat];
    }
    else if (securityLevel == 192 ){
        return &parameters[2][keyFormat];
    }
    else if (securityLevel == 256 ){
        return &parameters[3][keyFormat];
    }
}
    