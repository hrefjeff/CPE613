#include <Timer.hpp>

#include <cuda_runtime.h>
#include <helper_cuda.h>

class Timer::Implementation {
public:
  Implementation () {
    
    checkCudaErrors (
      cudaEventCreate (
        &start_
      )
    );

    checkCudaErrors (
      cudaEventCreate (
        &stop_
      )
    );

    // this is done so that all future calls to elapsed will succeed
    start();
    stop();

  }

  ~Implementation() {
    
    checkCudaErrors (
      cudaEventDestroy (
        start_
      )
    );
    
    checkCudaErrors (
      cudaEventDestroy (
        stop_
      )
    );
  
  }

  // Delete copy constructor and assignment operator
  Implementation (Implementation const &) = delete;
  Implementation & operator= (Implementation const &) = delete;
  // Not technically required due to the previous lines
  Implementation (Implementation &&) = delete;
  Implementation & operator= (Implementation &&) = delete;

  void start() {
    checkCudaErrors (
      cudaEventRecord (
        start_
      )
    );  
  }

  void stop() {
    
    checkCudaErrors (
      cudaEventRecord (
        stop_
      )
    );

  }

  float elapsedTime_ms () {
    
    checkCudaErrors (
      cudaEventSynchronize (
        stop_
      )
    );

    float milliseconds = 0.0f;

    checkCudaErrors (
      cudaEventElapsedTime (
        &milliseconds,
        start_,
        stop_
      )
    );

    return milliseconds;

  }

private:
  
  cudaEvent_t start_;
  cudaEvent_t stop_;

};

Timer::Timer () : 
  implementation_ (
    std::make_shared<Implementation>()
  ) {}

void Timer::start() {

  implementation_->start();

}

void Timer::stop() {

  implementation_->stop();

}

float Timer::elapsedTime_ms() {

  return implementation_->elapsedTime_ms();

}
