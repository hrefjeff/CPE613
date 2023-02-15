# 1. "Explain, in your own words (and preferably in a more straightforward way than the articles I've found online, though you can use their discussion as a starting place), how thread scheduling is done on the Volta architecture or later. Include details on any architectural changes / paradigm shifts that were necessary to make this happen."

Independent thread scheduling was made possible mainly because the Volta and later models were transformed to allow each thread to maintain it's own state.
Each thread has been given it's own call stack and program counter which allows threads to diverge and reconverge at a sub-warp (less than 32 thread group) level.



# 2. "Can you walk me through a code example to illustrate how this will benefit a particular kernel's execution?"

```c++
if (threadIdx.x < 4)
  A;
  B;
} else {
  X;
  Y;
}
Z;
```


# 3. "Is the performance backward compatible? That is, if we run our code written and optimized for pre-Volta, are we going to lose performance due to architectural changes? Demonstrate to me why this is or isn't the case."

The code written in CUDA does not change. The underlying system calls however are treated differently. This is the benefit of well written interfaces that users of CUDA can benefit from.

# 4. (10 pts extra credit) Once you've answered "your manager's questions", see if you can justify your claims in reality by running some code examples on the Pascal, Volta, and Ampere (post-Volta) GPUs provided by the DMC

