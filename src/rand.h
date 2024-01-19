#include <cstdlib>
#include <ctime>

class curandState
{
public:
    inline curandState(int seed) { std::srand(std::time(NULL) + seed); }

    inline float randomUnit() { return (float)std::rand() / (float)RAND_MAX; }
};
