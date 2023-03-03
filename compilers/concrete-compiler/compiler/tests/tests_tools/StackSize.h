#ifndef TEST_TOOLS_STACK_SIZE_H
#define TEST_TOOLS_STACK_SIZE_H

#include <iostream>
#include <sys/resource.h>

void setCurrentStackLimit(size_t size) {
  rlim_t stack_size_target = size;
  struct rlimit rl;

  int res = getrlimit(RLIMIT_STACK, &rl);
  if (!res) {
    if (rl.rlim_cur < stack_size_target) {
      rl.rlim_cur = stack_size_target;
      res = setrlimit(RLIMIT_STACK, &rl);
      if (res)
        std::cerr << "Unable to set the requiresd stack size ("
                  << stack_size_target << " Bytes) - setrlimit returned " << res
                  << std::endl;
      else
        std::cerr << "Stack limit increased to " << rl.rlim_cur << " Bytes\n";
    }
  }
}

#endif
