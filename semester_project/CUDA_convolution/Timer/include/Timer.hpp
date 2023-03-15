#ifndef __timer_hpp
#define __timer_hpp

#include <memory>

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

#endif


