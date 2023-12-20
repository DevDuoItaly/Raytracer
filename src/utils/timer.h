#pragma once

#include <chrono>

class Timer
{
public:
    inline Timer() { Reset(); }
    
    inline void Reset() { m_Start = clock_::now(); }

    inline double ElapsedMillis() const
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(clock_::now() - m_Start).count()
            * 0.001f * 0.001f;
    }

private:
    typedef std::chrono::high_resolution_clock clock_;
    std::chrono::time_point<clock_> m_Start;
};
