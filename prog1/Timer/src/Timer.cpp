#include <Timer.hpp>

class Timer {
  
  public:
    Timer();
    void start();
    void stop();
    float elapsedTime_ms();
    
  private:
    class Implementation;
    std::shared_ptr<Implementation> implementation_;

};

Timer::start()
