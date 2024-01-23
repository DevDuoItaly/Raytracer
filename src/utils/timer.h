#pragma once

#include <chrono>

class Timer
{
public:
    inline Timer() { Reset(); }
    
    // Resets the timer by setting the start point to the current time
    inline void Reset() { m_Start = clock_::now(); }

    // Calculates and returns the elapsed time in milliseconds
    inline double ElapsedMillis() const
    {
        // Calculates the difference between the current time and the start point in nanoseconds
        // and converts the result from nanoseconds to milliseconds
        return std::chrono::duration_cast<std::chrono::nanoseconds>(clock_::now() - m_Start).count()
            * 0.001f * 0.001f;
    }

private:
    typedef std::chrono::high_resolution_clock clock_;
    std::chrono::time_point<clock_> m_Start;
};
